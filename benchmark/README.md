# Performance analysis

The original implementation of our dynamic sparseness (`gmul/`) is slower than cuBLAS matrix multiplication. However, recently google release a
sparse kernels for SDDMM and SPMM [here](https://github.com/google-research/google-research/tree/master/sgk) which
performs pretty good. We adopted their code and use it for block dynamic sparseness benchmark.

## How to run?

Install [sgk](https://github.com/google-research/google-research/tree/master/sgk) kernels for sparse matrix
multiplication and run:

``python benchmark_lstm.py --config sparse``   


