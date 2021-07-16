import torch
from torch import nn
from torch.autograd import gradcheck
from tcop.mm import MyLinearFunc
import time

def check_gradient():
    # sequence_length, dim_input, dim_output = 4, 4, 4
    # sequence_length, dim_input, dim_output = 128, 128, 128
    sequence_length, dim_input, dim_output = 64, 64, 64
    X = torch.DoubleTensor(sequence_length, dim_input).cuda().normal_()
    W = torch.DoubleTensor(dim_input, dim_output).cuda().normal_()

    X = nn.Parameter(X)
    W = nn.Parameter(W)

    print('gradcheck:', gradcheck(MyLinearFunc.apply, [X, W]))


start = time.time()
check_gradient()
print("time:", time.time()-start)
