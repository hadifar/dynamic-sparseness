import torch
import time

import lltm_cuda


def gmv_forward(input, weight, gate, version):
    output = torch.zeros(input.size(0), weight.size(1), dtype=input.dtype).cuda()
    lltm_cuda.gmv_forward_gpu(output, input, weight, gate, version)
    return output


def gmv_backward_w(input, doutput, gate, version):
    dweight = torch.zeros(input.size(1), doutput.size(1), dtype=input.dtype).cuda()
    lltm_cuda.gmv_backward_w_gpu(doutput, input, dweight, gate, version)
    return dweight


def gmv_backward_g(input, doutput, weight, block_size, version):
    dgate = torch.zeros(input.size(0), int(input.size(1)/block_size), int(doutput.size(1)/block_size), dtype=input.dtype).cuda()
    lltm_cuda.gmv_backward_g_gpu(doutput, input, weight, dgate, version)
    return dgate


# gmv_version = 4
# gmv_version = 5
gmv_version = 6


class GMV(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, gate):
        ctx.save_for_backward(input, weight, gate)
        output = torch.zeros(input.size(0), weight.size(1), dtype=input.dtype).cuda()
        # print("forward output:{} input:{} weight:{} gate:{}".format(output.size(), input.size(), weight.size(), gate.size()))
        start = time.time()
#        lltm_cuda.gmm_forward_cpu(output, input, weight, gate)
        lltm_cuda.gmv_forward_gpu(output, input, weight, gate, gmv_version)
        # print("->", output.sum().item())
        # print("forward:", time.time()-start)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, gates = ctx.saved_tensors
        grad_input = grad_weights = grad_gates = None

        if ctx.needs_input_grad[0]:
           grad_input = torch.zeros_like(input)
           # print("grad1 dinput:{} doutput:{} weight:{} gate:{}".format(grad_input.size(), grad_output.size(), weights.t().size(), gates.permute(0,2,1).size()))
           start = time.time()
#           lltm_cuda.gmm_forward_cpu(grad_input, grad_output.contiguous(), weights.t().contiguous(), gates.permute(0,2,1).contiguous())
           lltm_cuda.gmv_forward_gpu(grad_input, grad_output.contiguous(), weights.t().contiguous(), gates.permute(0,2,1).contiguous(), gmv_version)
           # print("->", grad_input.sum().item())
           # print("dx:", time.time()-start)


        if ctx.needs_input_grad[1]:
           grad_weights = torch.zeros_like(weights)
           # print("grad1 dweights:{} input:{} doutput:{} gate:{}".format(grad_weights.size(), input.size(), grad_output.size(), gates.size()))
           start = time.time()
#           lltm_cuda.gmm_backward_w_cpu(grad_weights, input, grad_output.contiguous(), gates)
           lltm_cuda.gmv_backward_w_gpu(grad_output.contiguous(), input.contiguous(), grad_weights, gates, gmv_version)
           # print("->", grad_weights.sum().item())
           # print("dw:", time.time()-start)
        if ctx.needs_input_grad[2]:
           grad_gates = torch.zeros_like(gates)
           # print("need grad2: gate:{} input:{} weight:{} doutput:{}".format(grad_gates.size(), input.size(), weights.size(), grad_output.size()))
           start = time.time()
#           lltm_cuda.gmm_backward_g_cpu(grad_gates, input, weights, grad_output.contiguous())
           lltm_cuda.gmv_backward_g_gpu(grad_output.contiguous(), input, weights, grad_gates, 7)
           # print("->", grad_gates.sum().item())
           # print("dg:", time.time()-start)

        return grad_input, grad_weights, grad_gates