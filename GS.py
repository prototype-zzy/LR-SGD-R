import argparse
import random
import sys
import time
import math
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.onnx
import torch.optim as optim
from torch.multiprocessing import Process
from math import ceil


from data import Corpus
import model as m
from utils import batchify, repackage_hidden, get_batch, TensorBuffer
from timer import *


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=650,   # 200
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=650,     # 200
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,    # 2
                    help='number of layers')
parser.add_argument('--lr', type=float, default=12.5,    # 20
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,  # 40
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',   # 20
                    help='batch size')
parser.add_argument('--bptt', type=int, default=30,  # 35
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,  # 0.2
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true', help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt',  help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='', help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dryrun', action='store_true', default=False, help='verify the code and the model')
parser.add_argument('--backend', type=str, default='nccl', help='gloo or nccl')
parser.add_argument('--workers', type=int, default=2, help='number of workers')
parser.add_argument('--gpu', type=list, default=[0, 1, 0, 0], help='define gpu devices')
parser.add_argument('--compress', type=float, default=0.1, help='compression rate')
parser.add_argument('--reducer', type=str, default='topk', help='use topk or randomk as reducer')
args = parser.parse_args()


class Reducer:
    def __init__(self, device):
        self.rng = np.random.RandomState(args.seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device

    def reduce(self, grad_in, grad_out):
        """Return communicated bits"""
        raise NotImplementedError()

class RandomKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, device, timer, compression=1 / 244):
        super().__init__(device)
        self.timer = timer
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out, epoch):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("prepare grad size", epoch):

            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]                                                            # 记录每一层tensor的开始位置
            for tensor in grad_in:
                top_size = max(1, int(self.compression * tensor.nelement()))            # 最大取数数量 max(压缩率 * 一层参数梯度tensor的元素数量)
                flatgrad_size += top_size                                               # 最大数量累加
                tensor_idx.append(tensor_idx[-1] + top_size)                            # 记录每一层的开始位置
            flatgrad_start_idx = tensor_idx[:-1]                                        # 展平后起始位置合集: 0 ~ n-1
            flatgrad_end_idx = tensor_idx[1:]                                           # 展平后结束位置合集: 1 ~ n
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("prepare grad value", epoch):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(self.compression * tensor.nelement()))
                with self.timer("pv.select random", epoch):
                    positions = torch.tensor(random.sample(range(tensor.nelement()), top_size))
                    # _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)    # 取tenor中绝对值最大的top_size个的位置，放入positions中
                with self.timer("pv.record value and pos", epoch):
                    values = tensor.view(-1)[positions].contiguous()                            # 把值放进values中
                    flat_values[start:end] = values
                    flat_positions[start:end] = positions

            with self.timer("pv.process grad and mem", epoch):
                for tensor, mem, start, end in zip(grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx):
                    positions = flat_positions[start:end]
                    mem.data[:] = tensor
                    mem.view(-1)[positions.long()] = 0.0        # 同步过的梯度置0

        with self.timer("all gather", epoch):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]

        with self.timer("average", epoch):

            for tensor, out, start, end in zip(grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers

class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, device, timer, compression=1 / 244):
        super().__init__(device)
        self.timer = timer
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out, epoch):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("prepare grad size", epoch):

            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]                                                            # 记录每一层tensor的开始位置
            for tensor in grad_in:
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))      # 最大取数数量 max(0.5 * 压缩率 * 一层参数梯度tensor的元素数量)
                flatgrad_size += top_size                                               # 最大数量累加
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]                                        # 展平后起始位置合集: 0 ~ n-1
            flatgrad_end_idx = tensor_idx[1:]                                           # 展平后结束位置合集: 1 ~ n
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("prepare grad value", epoch):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(0.5 * self.compression * tensor.nelement()))
                with self.timer("select top", epoch):
                    _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)    # 取tenor中绝对值最大的top_size个的位置，放入positions中
                values = tensor.view(-1)[positions].contiguous()                            # 把值放进values中
                flat_values[start:end] = values
                flat_positions[start:end] = positions

            for tensor, mem, start, end in zip(
                    grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0        # 同步过的梯度置0

        with self.timer("all gather", epoch):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]

        with self.timer("average", epoch):

            for tensor, out, start, end in zip(grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


def evaluate(data_source, model, corpus, criterion, eval_batch_size=1):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)



def run(rank, size):
    log = open("log"+args.reducer+str(rank)+".txt", "a")
    print(args, file=log)
    torch.manual_seed(args.seed)

    # 设置device
    device = torch.device("cuda:{}".format(args.gpu[rank]) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu[rank])
    print(device)

    # 处理数据集
    corpus = Corpus(args.data)

    eval_batch_size = 1
    train_data = batchify(corpus.train, args.batch_size, device, rank, dist.get_world_size(), 1)
    val_data = batchify(corpus.valid, eval_batch_size, device, rank, dist.get_world_size())
    test_data = batchify(corpus.test, eval_batch_size, device, rank, dist.get_world_size())


    # 生成模型
    ntokens = len(corpus.dictionary)
    model = m.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(
            device) if args.model == 'Transformer' else m.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(
            device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr * args.workers)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(args.epochs * 0.833)], gamma=0.1)   # 学习率调整
    criterion = nn.CrossEntropyLoss()
    num_batches = ceil(len(train_data) / float(args.batch_size))

    timer = Timer(rank)

    if args.reducer == 'topk':
        reducer = TopKReducer(device, timer, args.compress)
    else:
        # reducer = UniformRandomSparseReducer(device, timer, args.compress)
        reducer = RandomKReducer(device, timer, args.compress)

    memories = [torch.zeros_like(param).to(device) for param in model.parameters()]  # 误差项
    send_buffers = [torch.zeros_like(param).to(device) for param in model.parameters()]  # 计算时的缓存

    # 开始训练
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.
        ntokens = len(corpus.dictionary)
        if args.model != 'Transformer':
            hidden = model.init_hidden(args.batch_size)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
            with timer("forward & backward", epoch):
                data, targets = get_batch(train_data, i, args.bptt)
                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                model.zero_grad()
                if args.model == 'Transformer':
                    output = model(data)
                    output = output.view(-1, ntokens)
                else:
                    hidden = repackage_hidden(hidden)
                    output, hidden = model(data, hidden)
                loss = criterion(output, targets)
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                # print(rank, loss.item)

            with timer("prepare memory"):
                grads = [param.grad.data for param in model.parameters()]
                for grad, memory, send_bfr in zip(grads, memories, send_buffers):
                    send_bfr.data[:] = grad + memory
            reducer.reduce(send_buffers, grads, memories, epoch)  # 输入,输出,误差项

            with timer("optimizer step", epoch):
                optimizer.step()

        scheduler.step()
        if args.dryrun:
            cur_loss = total_loss / num_batches
            print('Rank ', dist.get_rank(), ', epoch ', epoch, ': ', cur_loss, 'lr:',
                optimizer.state_dict()['param_groups'][0]['lr'])

            # 验证集测试
            val_loss = evaluate(test_data, model, corpus, criterion)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
            print(epoch, val_loss, math.exp(val_loss), file=log)

    # 结束训练
    # 测试集测试
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()
    test_loss = evaluate(test_data, model, corpus, criterion, 1)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))

    if rank == 0:
        print(timer.summary())
    print(timer.summary(), file=log)
    log.close()


def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = args.workers
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run, args.backend))
        p.start()
        processes.append(p)

    for p in processes:
        # print(p.name)
        p.join()
