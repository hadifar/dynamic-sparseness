import torch
from torch import nn
from torch.autograd import gradcheck
import time

from tcop.mm import my_matmul_nn, my_matmul_tn


batch, dim_input, dim_output = 44*32, 1536, 4608
blocksize = 128
X = torch.Tensor(batch, dim_input).cuda().normal_()
W = torch.Tensor(dim_input, dim_output).cuda().normal_()

# X.fill_(0)
# W.fill_(0)
#
# X[0, :] = 1
# W[0, :] = 1

runs = 10

for run in range(runs):
    start = time.time()
    output1 = my_matmul_nn(X, W, ref=False)

    sum1 = output1.sum()
    print('{}\toutput1: {}'.format(time.time()-start, sum1));

    start = time.time()
    output2 = my_matmul_nn(X, W, ref=True)
    sum2 = output2.sum()
    print('{}\toutput2: {}'.format(time.time()-start, sum2));

    print("diff:", (output1-output2).abs().sum().item())

# print(output1[0:8,0:8])
# print(output2[0:8,0:8])

# for i in [0, 4, 16, 32, 64, 128, 256]:
# for i in [32]:
#     print(i, (output1[0:i,0:i]-output2[0:i,0:i]).abs().sum().item())
#     # print(output1[0:i*2,0:i]-output2[0:i*2,0:i])
#     print(output1[0:i*2,0:i])

# print(output1-output2)

print(output1)
print(output2)