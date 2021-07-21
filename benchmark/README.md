# Performance analysis

The original implementation of our dynamic sparseness (`gmul/`) is slower than cuBLAS matrix multiplication. However, google recently published
sparse kernels for SDDMM and SPMM [here](https://github.com/google-research/google-research/tree/master/sgk) which have 
reasonable performance. We adopted their code and use it for block dynamic sparseness benchmark.

Note that we exclude the cost of dense-to-sparse & sprase-to-dense operations (by-passing the operations) since these two operations become bottleneck for matrix-multiplication with the given API.

## How to run?

- Install [sgk](https://github.com/google-research/google-research/tree/master/sgk) for sparse matrix
multiplication and run:

- ``python benchmark_lstm.py --config sparse``   

