import torch
from torch import nn
from torch.autograd import gradcheck
from tcop.gmv import GMV
import time

def check_gradient():
    # sequence_length, dim_input, dim_output, block_size = 128, 128, 128, 128
    # sequence_length, dim_input, dim_output, block_size = 128, 128, 256, 128
    sequence_length, dim_input, dim_output, block_size = 128, 256, 128, 128
    X = torch.DoubleTensor(sequence_length, dim_input).cuda().normal_()
    W = torch.DoubleTensor(dim_input, dim_output).cuda().normal_()
    G = torch.DoubleTensor(sequence_length, int(dim_input/block_size), int(dim_output/block_size)).cuda().normal_()

    # X = nn.Parameter(X)
    # G = nn.Parameter(G)
    W = nn.Parameter(W)

    print('gradcheck:', gradcheck(GMV.apply, [X, W, G]))


start = time.time()
check_gradient()
print('time:', time.time()-start)