import torch
from torch import nn
from torch.autograd import gradcheck
from tcop.gmm import GMM

def check_gradient():
    batch = 2
    rows = 8
    cols = 12
    width = 16
    blocksize = 4
    X = torch.DoubleTensor(batch, rows, width).cuda().normal_()
    W = torch.DoubleTensor(width, cols).cuda().normal_()
    G = torch.DoubleTensor(batch, int(width/blocksize), int(cols/blocksize)).cuda().normal_()

    X = nn.Parameter(X)
    G = nn.Parameter(G)
    W = nn.Parameter(W)

    print('gradcheck:', gradcheck(GMM.apply, [X, W, G]))

check_gradient()