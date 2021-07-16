import math
import torch
from torch import nn
from torch.nn import Parameter, init

import lltm_cuda

def mymm(A, B):
   C = torch.zeros(A.size()[0], B.size()[1]).cuda()
   lltm_cuda.mymm4(C, A, B)
   return C

def mymatmul(A, B):
#   print("A:", A.shape)
#   print("B:", B.shape)
   if A.dim() == 3:
      b, rows, cols = A.size()
      C = torch.zeros(b * rows, B.size()[1]).cuda()
      lltm_cuda.mymm4(C, A.view(-1, cols), B)
      return C.view(b, rows, -1)
   elif A.dim() == 2:
      return mymm(A, B)

class MyLinearFunc(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
#        outputs1 = mymm(input, weight.t().contiguous())
#        print("input:", input.shape)
#        print("weight:", weight.shape)
#        outputs2 = input.mm(weight.t())
#        output = torch.matmul(input, weight.t())
        output = mymatmul(input, weight.t().contiguous())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
#        tmp = (outputs1-outputs2).sum()
#        print(tmp.item())
#        if abs(tmp) > 1e-4:
#           print(outputs1)
#           print(outputs2)
#           exit(1)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
#            grad_input = grad_output.mm(weight)
#            grad_input = torch.matmul(grad_output, weight)
             grad_input = mymatmul(grad_output, weight)
        if ctx.needs_input_grad[1]:
#            grad_weight = grad_output.t().mm(input)
            tmp_doutput = grad_output.view(-1, grad_output.size()[-1])
            tmp_input = input.view(-1, input.size()[-1])
#            print("grad_output:", tmp_doutput.shape)
#            print("input:", tmp_input.shape)
#            grad_weight = tmp_doutput.t().mm(tmp_input)
            grad_weight = mymm(tmp_doutput.t().contiguous(), tmp_input)
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
        return MyLinearFunc.apply(input, self.weight, self.bias)
