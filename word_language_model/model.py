import math

import torch
import torch.nn as nn

try:
    from tcop.gmv import GMV
except:
    print('Cannot import GMV')
    pass


class Sparsify1D(nn.Module):
    def __init__(self, sparse_ratio=0.5):
        super().__init__()
        self.sr = sparse_ratio
        self.preact = None
        self.act = None

    def forward(self, x):
        k = int(math.ceil(self.sr * x.shape[1]))

        topval = x.topk(k, dim=1)[0][:, -1]
        topval = topval.expand(x.shape[1], x.shape[0]).permute(1, 0)
        comp = (x >= topval).to(x)
        res = comp * x

        res = res / (torch.sum(res, dim=-1, keepdim=True) / x.shape[-1])

        return res

    def get_activation(self):
        def hook(model, input, output):
            self.preact = input[0].cpu().detach().clone()
            self.act = output.cpu().detach().clone()

        return hook

    def record_activation(self):
        self.register_forward_hook(self.get_activation())


class RNNCellBase(nn.Module):
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias, num_chunks):
        super(RNNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(input_size, num_chunks * hidden_size, ))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, num_chunks * hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

        # nn.RNNCell(3, 5)

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)


class GatedLSTMCell(RNNCellBase):

    def __init__(self, input_size, hidden_size, bias=True, blocksize=128, sparsity=0.5, mode='sparse'):
        super(GatedLSTMCell, self).__init__(input_size, hidden_size, bias, num_chunks=4)
        self.block_size = blocksize

        if mode == 'torch':
            self.ih_nblock = int((input_size * hidden_size * 4) / (blocksize * blocksize))
            self.hh_nblock = int((hidden_size * hidden_size * 4) / (blocksize * blocksize))
            self.g_ih = nn.Sequential(nn.Linear(input_size, self.ih_nblock), Sparsify1D(sparsity))
            self.g_hh = nn.Sequential(nn.Linear(hidden_size, self.hh_nblock), Sparsify1D(sparsity))
        elif mode == 'static_block':
            # todo: make it clean
            self.ih_nblock = int(self.input_size / self.block_size)
            self.hh_nblock = int(self.hidden_size / self.block_size)

            if sparsity == 0.5:
                block = torch.ones((int(input_size * (1 / 2)), int(self.ih_nblock * self.hh_nblock * 4 * (1 / 2))),
                                   dtype=torch.float32)
                self.g_ih = torch.block_diag(block, block)
                self.g_hh = torch.block_diag(block, block)
            elif sparsity == 0.75:
                block = torch.ones((int(input_size * (1 / 4)), int(self.ih_nblock * self.hh_nblock * 4 * (1 / 4))),
                                   dtype=torch.float32)
                self.g_ih = torch.block_diag(block, block, block, block)
                self.g_hh = torch.block_diag(block, block, block, block)
            elif sparsity == 0.9:
                block = torch.ones((int(input_size * (1 / 10)), int(self.ih_nblock * self.hh_nblock * 4 * (1 / 10))),
                                   dtype=torch.float32)
                self.g_ih = torch.block_diag(block, block, block, block, block, block, block, block, block, block)
                self.g_hh = torch.block_diag(block, block, block, block, block, block, block, block, block, block)
        else:
            self.ih_nblock = int(self.input_size / self.block_size)
            self.hh_nblock = int(self.hidden_size / self.block_size)
            self.g_ih = nn.Sequential(nn.Linear(input_size, self.ih_nblock * self.hh_nblock * 4),
                                      Sparsify1D(sparsity))
            self.g_hh = nn.Sequential(nn.Linear(hidden_size, self.hh_nblock * self.hh_nblock * 4),
                                      Sparsify1D(sparsity))

        self.epoch = 1

    def blockmul1(self, inp, hx, weight_ih, weight_hh):
        """this will not work for large hidden sizes"""
        gate1 = self.g_ih(inp)
        gate2 = self.g_hh(hx)
        wshape = weight_ih.shape
        new_weight_ih = weight_ih.view(-1, self.block_size, self.block_size)
        new_weight_hh = weight_hh.view(-1, self.block_size, self.block_size)
        res = []
        for i0, i1, g1, g2 in zip(inp, hx, gate1, gate2):
            res += [
                i0 @ (g1.unsqueeze(-1).unsqueeze(-1) * new_weight_ih).view(wshape)
                +
                i1 @ (g2.unsqueeze(-1).unsqueeze(-1) * new_weight_hh).view(wshape)
            ]

        return torch.stack(res, 0)

    def blockmul2(self, inp, hx, weight_ih, weight_hh):
        bsize = inp.shape[0]
        gih = self.g_ih(inp).view(bsize, self.ih_nblock, 4 * self.hh_nblock)
        ghh = self.g_hh(hx).view(bsize, self.hh_nblock, 4 * self.hh_nblock)
        g1 = GMV.apply(inp, weight_ih, gih).view(bsize, -1)
        g2 = GMV.apply(hx, weight_hh, ghh).view(bsize, -1)
        return g1 + g2

    def blockmul3(self, inp, hx, weight_ih, weight_hh):
        bsize = inp.shape[0]
        gih = self.g_ih.view(bsize, self.ih_nblock, 4 * self.hh_nblock)
        ghh = self.g_hh.view(bsize, self.hh_nblock, 4 * self.hh_nblock)
        g1 = GMV.apply(inp, weight_ih, gih).view(bsize, -1)
        g2 = GMV.apply(hx, weight_hh, ghh).view(bsize, -1)
        return g1 + g2

    def forward(self, input, hx=None):
        # type: # (Tensor, Optional[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]
        self.check_forward_input(input)

        if hx is None:
            zeros = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
            hx = (zeros, zeros)

        hx, cx = hx

        gates = self.blockmul2(input, hx, self.weight_ih, self.weight_hh) + self.bias_ih + self.bias_hh

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)

        hy = outgate * torch.tanh(cy)

        return hy, cy


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=True, sparsity=0.5,
                 blocksize=128):

        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        rnn_modulelist = []
        for i in range(nlayers):
            if rnn_type == 'GatedLSTMCell' or rnn_type == 'GatedLSTM':
                rnn_modulelist.append(GatedLSTMCell(ninp, nhid, sparsity=sparsity, blocksize=blocksize))
            elif rnn_type in ['LSTMCell', 'GRUCell']:
                rnn_modulelist.append(getattr(nn, rnn_type)(ninp, nhid))
            else:
                raise ValueError("(RNNCell) is not implemented...")

            if i < nlayers - 1:
                rnn_modulelist.append(nn.Dropout(dropout))

        self.rnn = nn.ModuleList(rnn_modulelist)

        self.decoder = nn.Linear(nhid, ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        layer_input = emb
        new_hidden = [[], []]
        for idx_layer in range(0, self.nlayers + 1, 2):
            output = []
            hx, cx = hidden[0][int(idx_layer / 2)], hidden[1][int(idx_layer / 2)]
            for idx_step in range(input.shape[0]):
                hx, cx = self.rnn[idx_layer](layer_input[idx_step], (hx, cx))
                output.append(hx)
            output = torch.stack(output)
            if idx_layer + 1 < self.nlayers:
                output = self.rnn[idx_layer + 1](output)
            layer_input = output
            new_hidden[0].append(hx)
            new_hidden[1].append(cx)
        new_hidden[0] = torch.stack(new_hidden[0])
        new_hidden[1] = torch.stack(new_hidden[1])
        hidden = tuple(new_hidden)

        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type in ['LSTM', 'LSTMCell', 'GatedLSTMCell', 'GatedLSTM']:
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

    def on_epoch_begin(self, epoch):
        for rnn in self.rnn:
            if hasattr(rnn, 'on_epoch_begin'):
                rnn.on_epoch_begin(epoch)

        print('*' * 50)
        print('whats going on man')
        print('*' * 50)
