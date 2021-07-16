import math
import torch
from torch import nn
from tcop.gmm import GMM

torch.manual_seed(2019)


class TestGradient(torch.nn.Module):
    def __init__(self, gradX=True, gradW=True, gradG=True):
        super(TestGradient, self).__init__()
        batch, rows, cols, width = 2, 70, 1536, 4608
        # batch, rows, cols, width = 2, 70, 512, 1024
#        batch, rows, cols, width = 2, 8, 12, 16
        blocksize = 4
        X = torch.DoubleTensor(batch, rows, width).cuda().normal_()
        W = torch.DoubleTensor(width, cols).cuda().normal_()
        G = torch.DoubleTensor(batch, int(width/blocksize), int(cols/blocksize)).cuda().normal_()

        self.X = nn.Parameter(X) if gradX else X
        self.W = nn.Parameter(W) if gradW else W
        self.G = nn.Parameter(G) if gradG else G

    def forward(self):
        Y = GMM.apply(self.X, self.W, self.G)
        sum = Y.sum()
        return sum * sum


def test_opt_lbfgs():
   model = TestGradient()

   #optimizer = torch.optim.Adam(model.parameters())
   optimizer = torch.optim.LBFGS(model.parameters())

   iters = 100
   for iter in range(iters):
      def closure():
         optimizer.zero_grad()
         loss = model()
         print("->", loss.item())
         loss.backward()
         return loss
      optimizer.step(closure)

def test_opt_adam():
   model = TestGradient()

   optimizer = torch.optim.Adam(model.parameters())

   iters = 100
   for iter in range(iters):
      optimizer.zero_grad()
      loss = model()
      loss.backward()
      optimizer.step()

      print("iter {}: {}".format(iter, loss.item()))




test_opt_adam()
