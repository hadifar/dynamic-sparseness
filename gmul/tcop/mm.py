import math
import torch
from torch import nn
from torch.nn import Parameter, init

import lltm_cuda


def my_matmul(A, B, ref=False):
    if ref:
        return torch.matmul(A, B)
    else:
        C = torch.zeros(A.size()[0], B.size()[1], dtype=A.dtype).cuda()
        # print(C.size(), A.size(), B.size())
        lltm_cuda.mm5(C, A.contiguous(), B.contiguous())
        return C


def my_matmul_nn(A, B, ref=False):
    if ref:
        return torch.matmul(A, B)
    else:
        C = torch.zeros(A.size()[0], B.size()[1], dtype=A.dtype).cuda()
        # print(C.size(), A.size(), B.size())
        lltm_cuda.mm5(C, A, B)
        return C

        # A0 = torch.zeros(128, 128, dtype=A.dtype).cuda()
        # B0 = torch.zeros(128, 128, dtype=B.dtype).cuda()
        # C0 = torch.zeros(128, 128, dtype=A.dtype).cuda()
        # A0[0:A.size()[0], 0:A.size()[1]] = A
        # B0[0:B.size()[0], 0:B.size()[1]] = B
        # lltm_cuda.mm5(C0, A0, B0)
        #
        # C = torch.zeros(A.size()[0], B.size()[1], dtype=C0.dtype).cuda()
        # C.copy_(C0[0:C.size()[0], 0:C.size()[1]])
        # return C


def my_matmul_tn(A, B, ref=False):
    if ref:
        return torch.matmul(A.t(), B)
    else:
        C = torch.zeros(A.size()[1], B.size()[1], dtype=A.dtype).cuda()
        # print(C.size(), A.size(), B.size())
        lltm_cuda.mm5_TN(C, A, B)
        return C


class MyLinearFunc(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = my_matmul_nn(input, weight)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = my_matmul(grad_output, weight.t())
            # grad_input = my_matmul_nn(weight, grad_output.t()).t()
        if ctx.needs_input_grad[1]:
            grad_weight = my_matmul(input.t(), grad_output)
            # grad_weight = my_matmul_tn(grad_output.t(), input).t()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias


class MyLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        seq, b, _ = input.shape
        input = input.view(seq * b, -1)
        output = MyLinearFunc.apply(input, self.weight.t(), self.bias)
        return output.view(seq, b, -1)
