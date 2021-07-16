from __future__ import print_function

import time
import argparse

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from tensorboard_logger import Logger as TBLogger
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms

from tcop.gmul import MyLinear

tb_logger = TBLogger('try')


class GatedBlock(nn.Module):
    def __init__(self, dim_input, dim_output, block_size, n_blocks, block_type='topk'):
        super(GatedBlock, self).__init__()
        self.block_size = block_size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_blocks = n_blocks
        self.block_dist_type = block_type

        # self.gates = nn.Linear(dim_input, n_blocks)
        self.gates = nn.Sequential(nn.Linear(dim_input, n_blocks), nn.Sigmoid())

        if block_type == 'mos':
            # self.prior_logit = nn.Linear(block_size, 10 * block_size)  # 10 is number of experts
            self.n_experts = 16
            self.prior_logit = nn.Sequential(nn.Linear(n_blocks, self.n_experts * n_blocks), nn.Tanh())
            # self.prior = nn.Linear(n_blocks, self.n_experts, bias=False)

        self.weight = Parameter(torch.Tensor(dim_input, dim_output))
        self.bias = Parameter(torch.Tensor(dim_output))

        self.T = 1
        self.K = 1

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b, _ = x.size()  # 64*512

        g = self.gates(x)  # 64*n_blocks
        # g = torch.softmax(g, -1)

        if self.block_dist_type == 'topk':
            pass
            # indexes of the smallest values
            _, a_ind = torch.topk(g, int(self.n_blocks / 2), sorted=False, largest=False)
            g = g.scatter(1, a_ind, 0)  # zero out the values of the smallest indexes
        elif self.block_dist_type == 'mos':
            g = self.prior_logit(g).view(b, self.n_experts, -1)
            t = (self.T - 1) + self.K / 938
            g = g / (1 / (t ** 2))
            self.K += 1
            # prior = self.prior(g)

            g = torch.nn.functional.softmax(g, -1)  # b*10,256
            # prior = torch.nn.functional.softmax(prior, -1).unsqueeze(-1)  # b*10*1

            g = g.sum(dim=1) * 4

            # print(50*'-')
            # print(g.data.cpu().numpy()[0])
            # print(50*'-')

            # self.K += 1
            # if self.K % 1000 == 0:
            #     print(g.data.cpu().numpy()[0])
            # g = torch.randint(high=2, size=[b, self.n_blocks],dtype=torch.float32)
            # g = torch.ones(size=[b,self.n_blocks])
            # g = torch.zeros(size=[b,self.n_blocks])

        else:
            raise NotImplementedError("Block selection method is not implemented...")

        g = g.unsqueeze(-1).unsqueeze(-1)  # 64*256*1

        weight0 = self.weight.view(self.n_blocks, self.block_size, self.block_size)

        # g = g.expand(b, self.n_blocks, self.block_size * self.block_size)  # 64*256*1024
        #
        # g = g.view(b,
        #            int(math.sqrt(self.n_blocks)),
        #            int(math.sqrt(self.n_blocks)),
        #            self.block_size,
        #            self.block_size)  # 64*256*1024 ==> 64*16*16*32*32
        #
        # # permutation
        # g = g.permute([0, 1, 3, 2, 4]).contiguous()
        #
        # # reshape gate to desire format
        # g = g.view(b, self.dim_output, self.dim_output)
        #
        # # weight0 = self.weight.unsqueeze(0)  # 1*512*512
        # # self.sample_weights = weight0.expand(b, -1, -1) * g  # b*512*512 . b*512*512

        g = g * weight0
        g = g.view(b, self.dim_output, self.dim_input)

        return torch.matmul(g, x.unsqueeze(-1)).squeeze(-1) + self.bias


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        # self.gb1 = GatedBlock(512, 512, block_size=64, n_blocks=64, block_type='topk')
        # self.gb2 = GatedBlock(512, 512, block_size=64, n_blocks=64, block_type='topk')
        # self.gb3 = GatedBlock(512, 512, block_size=64, n_blocks=64, block_type='topk')
        self.gb1 = MyLinear(512, 1024)
        self.gb2 = MyLinear(1024, 512)
        self.gb3 = MyLinear(512, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        b, _, _, _ = x.size()
        x = F.tanh(self.linear1(x.view(b, 784)))
        x = F.tanh(self.gb1(x))
        x = F.tanh(self.gb2(x))
        x = F.tanh(self.gb3(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def visualize(weights):
    print('#' * 50)
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(20, 20))
    columns = 8
    rows = 8
    for i in range(1, columns * rows + 1):
        img = weights[i - 1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray')
    plt.savefig('vis2.png')


def test(args, model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print('-' * 50)
#            if model.gb1.T == 1:
#                visualize(model.gc1.sample_weights.data.numpy())

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train(args, model, device, train_loader, optimizer, epoch)
        print("time:", time.time()-start)
        # todo: temperature
        # model.gb1.T = model.gb1.T + 1
        # model.gb2.T = model.gb2.T + 1
        # model.gb3.T = model.gb3.T + 1
        # model.gb1.K = 1
        # model.gb2.K = 1
        # model.gb3.K = 1

        # print('T value: ', model.gc1.T)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
