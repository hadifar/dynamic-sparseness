import torch

import time

from tcop.gmv import gmv_forward, gmv_backward_w, gmv_backward_g

block_size = 128


def rand_tensors():
    batch, dim_input, dim_output = 512, 256, 384
    # 512, 256, 384
    # batch, dim_input, dim_output = 44 * 32, 256, 512
    # batch, dim_input, dim_output = 1 * 1, 4, 4

    X = torch.Tensor(batch, dim_input).cuda().normal_()
    W = torch.Tensor(dim_input, dim_output).cuda().normal_()
    G = torch.Tensor(batch, int(dim_input / block_size), int(dim_output / block_size)).cuda().normal_()
    #
    # X = torch.Tensor(batch, dim_input).normal_()
    # W = torch.Tensor(dim_input, dim_output).normal_()
    # G = torch.Tensor(batch, int(dim_input / block_size), int(dim_output / block_size)).normal_()
    # G = torch.randint(high=2, size=[batch, int(dim_input / block_size), int(dim_output / block_size)]).float()

    return X, W, G


def refgmm(X, W, G):
    batch_size, _ = X.shape
    d_inp, d_out = W.shape
    n_blocks = int((d_inp * d_out) / (block_size ** 2))

    G = G.view(batch_size, -1)
    G = G.unsqueeze(-1)
    G = G.expand(batch_size, n_blocks, block_size * block_size)
    G = G.view(batch_size, int(d_inp / block_size), int(d_out / block_size), block_size, block_size)
    G = G.permute([0, 1, 3, 2, 4]).contiguous()
    G = G.view(batch_size, d_inp, d_out)

    weight0 = W.unsqueeze(0)
    G = weight0.expand(batch_size, -1, -1) * G

    return torch.matmul(X.unsqueeze(1), G).squeeze(1)


def test_fw():
    sequence_length, dim_input, dim_output, block_size = 512, 256, 384, 128
    X = torch.DoubleTensor(sequence_length, dim_input).cuda().normal_()
    W = torch.DoubleTensor(dim_input, dim_output).cuda().normal_()
    G = torch.DoubleTensor(sequence_length, int(dim_input/block_size), int(dim_output/block_size)).cuda().normal_()

    output1 = gmv_forward(X, W, G, 2)
    output2 = gmv_forward(X, W, G, 4)
    output3 = refgmm(X, W, G)

    print('diff-fw[1-2]:', (output1 - output2).abs().sum().item())
    print('diff-fw[2-3]:', (output2 - output3).abs().sum().item())

    print(output1)
    print(output2)
    print(output3)


def test_bw_w():
    # sequence_length, dim_input, dim_output, block_size = 512, 256, 384, 128
    sequence_length, dim_input, dim_output, block_size = 512, 256, 384, 128
    X = torch.DoubleTensor(sequence_length, dim_input).cuda().normal_()
    dY = torch.DoubleTensor(sequence_length, dim_output).cuda().normal_()
    G = torch.DoubleTensor(sequence_length, int(dim_input / block_size), int(dim_output / block_size)).cuda().normal_()

    # G.fill_(1.0)

    # dy = torch.eye(sequence_length).cuda()

    output0 = torch.matmul(X.t(), dY)
    output1 = gmv_backward_w(X, dY, G, 2)
    output2 = gmv_backward_w(X, dY, G, 4)

    print('diff-bw-w[1]:', (output0 - output1).abs().sum().item())
    print('diff-bw-w[2]:', (output0 - output2).abs().sum().item())
    print('diff-bw-w:', (output1 - output2).abs().sum().item())

    print(X.size())
    print(output0.size())
    print(output1.size())
    print(output2.size())


def test_bw_g():
    # sequence_length, dim_input, dim_output, block_size = 1024, 1024, 1024, 128
    sequence_length, dim_input, dim_output, block_size = 1536, 1536, 4608, 128
    # sequence_length, dim_input, dim_output, block_size = 128, 1024, 1024, 128
    # sequence_length, dim_input, dim_output, block_size = 1024, 1024, 1024, 64
    X = torch.DoubleTensor(sequence_length, dim_input).cuda().normal_()
    dY = torch.DoubleTensor(sequence_length, dim_output).cuda().normal_()
    W = torch.DoubleTensor(dim_input, dim_output).cuda().normal_()

    # X.fill_(0)
    # dY.fill_(0)
    # W.fill_(1)
    # W[0,0] = 2
    #
    # # X[:,0:256] = 1
    # # dY[:,0:256] = 1
    # X[:,0] = 1
    # dY[:,0] = 1

    # offset1 = 512
    # X[0,offset1:(offset1+128)] = 1
    # offset2 = 512
    # dY[0,offset2:(offset2+128)] = 1

    # X[:, 0] = 1
    # dY[:, 0] = 1

    for _ in range(10):
        output1 = gmv_backward_g(X, dY, W, block_size, 6)
        output2 = gmv_backward_g(X, dY, W, block_size, 7)

    print('diff-bw-g:', (output1 - output2).abs().sum().item())

    print(output1[0,:,:])
    print(output2[0,:,:])

    # print(output1.size())
    print(output2.size())


def blag():
    runs = 1
    for run in range(runs):
        X, W, G = rand_tensors()
        # print(G)
        # print(W)
        # print(X)
        start = time.time()
        output1 = gmv_forward(X, W, G)
        sum1 = output1.sum()
        print('{}\toutput1: {}'.format(time.time() - start, sum1))

        start = time.time()
        output2 = refgmm(X, W, G)
        # print(output2.shape)
        sum2 = output2.sum()
        print('{}\toutput2: {}'.format(time.time() - start, sum2))

        print("diff:", (output1 - output2).abs().sum().item())

# test_fw()
# test_bw_w()
test_bw_g()
