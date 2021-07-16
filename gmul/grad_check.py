from __future__ import division
from __future__ import print_function

import torch
from torch.autograd import gradcheck

from tcop.gmul import MyLinearFunc

cuda = True

device = torch.device("cuda")

kwargs = {'dtype': torch.double,
          'device': device,
          'requires_grad': True}

X = torch.randn(20, 10, 10, **kwargs)
W = torch.randn(10, 10, **kwargs)
variables = [X, W]

if gradcheck(MyLinearFunc.apply, variables):
    print('Ok')
