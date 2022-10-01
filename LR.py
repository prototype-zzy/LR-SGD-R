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
from utils import batchify, repackage_hidden, get_batch
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
args = parser.parse_args()


def generate_index(model, rank, num_batches, compression_rate, device, generator="rand"):
    start_time = time.time()
    length = args.epochs * num_batches
    indexs = []
    layers = 0
    for par in model.parameters():
        if par.ndim > 1:
            layers += 1

    for e in range(length):
        index_a_batch = torch.zeros(layers).to(device)
        if rank == 0:
            for i in range(layers):
                chunk_num = -1 * int(-1 // compression_rate)
                index_a_batch[i] = random.randint(0, chunk_num - 1)
        # print(index_a_batch)
        indexs.append(index_a_batch)

    for index in indexs:
        dist.all_reduce(index, dist.ReduceOp.SUM)
    # print("rank:", rank, indexs[0])
    print("generate time:", time.time()-start_time)
    return indexs


def average_weight(model, device, index, timer, epoch):
    index_list = index.tolist()
    for par in model.parameters():
        if par.ndim == 1:       # 一维参数全部同步
            with timer("reduce param & average", epoch):
                dist.all_reduce(par.data, dist.ReduceOp.SUM)
                par.data /= float(dist.get_world_size())
        else:                   # 多维参数按第二维度划分
            with timer("process", epoch):
                index = int(index_list.pop(0))
                chunk_size = int(par.data[0].nelement() * args.compress)
                # chunks = torch.split(par.data, chunk_size, dim=1)
                chunk = torch.split(par.data, chunk_size, dim=1)[index].contiguous().to(device)      # 按列划分后待同步的矩阵
            with timer("reduce param & average", epoch):
                dist.all_reduce(chunk, dist.ReduceOp.SUM)
                chunk /= float(dist.get_world_size())
            with timer("combine", epoch):
                par.data[:, index * chunk_size:min(((index + 1) * chunk_size), par.data[0].nelement())] = chunk   # 把值带回矩阵




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
    log = open("logLR"+str(rank)+".txt", "a")
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


    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=args.lr * args.workers)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs/2), int(args.epochs * 0.833)], gamma=0.1)   # 学习率调整
    criterion = nn.CrossEntropyLoss()
    num_batches = ceil(len(train_data) / float(args.bptt))

    indexs = generate_index(model, rank, num_batches, args.compress, device)
    timer = Timer(rank)

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

            with timer("optimizer step", epoch):
                optimizer.step()
            # 参数平均
            average_weight(model, device, indexs.pop(0), timer, epoch)

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

