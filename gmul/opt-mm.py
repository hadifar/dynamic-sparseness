import time
import torch
from torch import nn
from tcop.mm import MyLinearFunc

torch.manual_seed(2019)


class TestGradient(torch.nn.Module):
    def __init__(self, gradX=True, gradW=True, gradG=True):
        super(TestGradient, self).__init__()
        batch, dim_input, dim_output = 1536, 1536, 4608
        X = torch.Tensor(batch, dim_input).cuda().normal_()
        W = torch.Tensor(dim_input, dim_output).cuda().normal_()

        self.X = nn.Parameter(X) if gradX else X
        self.W = nn.Parameter(W) if gradW else W

    def forward(self):
        Y = MyLinearFunc.apply(self.X, self.W)
        sum = Y.sum()
        return sum * sum


def test_opt_adam():
    model = TestGradient()

    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters())

    start = time.time()

    iters = 100
    for iter in range(iters):
        optimizer.zero_grad()
        loss = model()
        loss.backward()
        optimizer.step()

        print("iter {}: {}".format(iter, loss.item()))

    print("time:", time.time()-start)


test_opt_adam()
