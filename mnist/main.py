from __future__ import print_function

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms

try:
    from tcop.gmv import GMV
except:
    print('GMV not loaded')
    pass


class Sparsify1D(nn.Module):
    def __init__(self, sparse_ratio=0.1):
        super().__init__()
        self.sr = sparse_ratio

    def forward(self, x):
        k = int(math.ceil(self.sr * x.shape[-1]))  # ceils since we at least need one block!

        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        res = comp * x

        res = res / (torch.sum(res, dim=-1, keepdim=True) / (x.shape[-1]))

        return res


class GatedBlock(nn.Module):
    def __init__(self, dim_input, dim_output, block_size, n_blocks, layer_name, sparsity_lvl):
        super(GatedBlock, self).__init__()
        self.layer_name = layer_name
        self.block_size = block_size
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.n_blocks = n_blocks

        self.sparsity_lvl = sparsity_lvl

        self.gates = nn.Sequential(nn.Linear(dim_input, n_blocks), Sparsify1D(sparsity_lvl))

        self.weight = Parameter(torch.Tensor(dim_input, dim_output))
        self.bias = Parameter(torch.Tensor(dim_output))

        self.T = 1

        self.K = 1

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # self.gates[0].bias.data.fill_(1)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        b_size, _ = x.size()
        self.sample_weight = g = self.gates(x)

        s_size = int(self.dim_input / self.block_size)
        g = g.view(-1, s_size, s_size)

        return GMV.apply(x, self.weight, g).view(b_size, -1) + self.bias


class Net(nn.Module):
    def __init__(self, moption, hidden_size, block_size, sparsity_lvl):
        super(Net, self).__init__()

        self.linear1 = nn.Linear(784, hidden_size)

        if moption == 'gated':
            n_blocks = int(hidden_size / block_size) ** 2
            self.gb1 = GatedBlock(hidden_size, hidden_size, block_size=block_size, n_blocks=n_blocks,
                                  layer_name='layer_1_',
                                  sparsity_lvl=sparsity_lvl)
            self.gb2 = GatedBlock(hidden_size, hidden_size, block_size=block_size, n_blocks=n_blocks,
                                  layer_name='layer_2_',
                                  sparsity_lvl=sparsity_lvl)
            self.gb3 = GatedBlock(hidden_size, hidden_size, block_size=block_size, n_blocks=n_blocks,
                                  layer_name='layer_3_',
                                  sparsity_lvl=sparsity_lvl)

            self.gb4 = GatedBlock(hidden_size, hidden_size, block_size=block_size, n_blocks=n_blocks,
                                  layer_name='layer_4_',
                                  sparsity_lvl=sparsity_lvl)
            self.gb5 = GatedBlock(hidden_size, hidden_size, block_size=block_size, n_blocks=n_blocks,
                                  layer_name='layer_5_',
                                  sparsity_lvl=sparsity_lvl)
        else:
            self.gb1 = nn.Linear(hidden_size, hidden_size)
            self.gb2 = nn.Linear(hidden_size, hidden_size)
            self.gb3 = nn.Linear(hidden_size, hidden_size)
            self.gb4 = nn.Linear(hidden_size, hidden_size)
            self.gb5 = nn.Linear(hidden_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        b, _, _, _ = x.size()
        x = F.relu(self.linear1(x.view(b, 784)))

        x = F.relu(self.gb1(x))
        x = F.relu(self.gb2(x))
        x = F.relu(self.gb3(x))
        x = F.relu(self.gb4(x))
        x = F.relu(self.gb5(x))

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


def test(args, model, device, test_loader, epoch):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')

    parser.add_argument('--name', type=str, default='experiment',
                        help='name of the exp')

    parser.add_argument('--moption', type=str, default='gated',
                        help='name of the exp', choices=['dense', 'gated'])

    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='size of the network')

    parser.add_argument('--block_size', type=int, default=128,
                        help='size of block')

    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='sparsity')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=654, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number')
    args = parser.parse_args()

    print(args)
    print('-' * 100)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(args.seed)

    # torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = torch.device("cuda:1")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=128, shuffle=True, drop_last=True, **kwargs)

    model = Net(moption=args.moption,
                hidden_size=args.hidden_size,
                block_size=args.block_size,
                sparsity_lvl=args.sparsity).to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)

        # todo: mnist works without sparsity scheduling
        # if hasattr(model.gb1, 'T'):
        #     model.gb1.T = model.gb1.T + 1
        #     model.gb2.T = model.gb2.T + 1
        #     model.gb3.T = model.gb3.T + 1
        #     model.gb4.T = model.gb3.T + 1
        #     model.gb5.T = model.gb3.T + 1

        test(args, model, device, test_loader, epoch)


if __name__ == '__main__':
    main()
