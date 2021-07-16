#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


#define     THREADS         256
#define     WARP_SIZE       32
#define     M               128
#define     N               128
#define     K               8
#define     THREAD_TILE_M   4
#define     THREAD_TILE_N   4

#define     Y0(a, b)        Y[(a)*y_cols + (b)]
#define     X0(a, b)        X[(a)*x_cols + (b)]
#define     W0(a, b)        W[(a)*w_cols + (b)]
#define     G0(a, b, c)     G[(a)*g_dim1*g_dim2 + (b)*g_dim2 + (c)]

#define     Xt(a, b)        X[a + (b)*x_cols]
#define     DY(a, b)        dY[(a)*y_cols + (b)]
#define     DW(a, b)        dW[(a)*w_cols + (b)]
#define     DG(a, b, c)     dG[(a)*g_dim1*g_dim2 + (b)*g_dim2 + (c)]


template<typename T> __global__ void gmv_128x128x8(T *Y, T *X, T *W, T *G, int rows, int cols, int x_cols, int g_dim1, int g_dim2) {
    const auto y_cols = cols;
    const auto w_cols = cols;
    const auto warp = threadIdx.x / WARP_SIZE;          // 8 warps
    const auto warp_row = warp / 4;                     // 2 (different order, does it matter?)
    const auto warp_col = warp % 4;                     // 4
    const auto thread = threadIdx.x % WARP_SIZE;        // 32 threads in a warp
    const auto thread_row = thread / 4;                 // 8 rows in a warp
    const auto thread_col = thread % 4;                 // 4 cols in a warp

    const auto shm_row = blockIdx.y * M;
    const auto shm_col = blockIdx.x * N;
    const auto row0 = shm_row + warp_row*128/2 + thread_row*THREAD_TILE_M;
    const auto col0 = shm_col + warp_col*128/4 + thread_col*THREAD_TILE_N;

    const auto w_rows = cols;
    const auto g_row = w_rows / g_dim1;
    const auto g_col = w_cols / g_dim2;         // 384 / 3 = 128

    T acc[2][2][THREAD_TILE_M][THREAD_TILE_N];

    // initialize 16x16 block of accumulators
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int a = 0; a < THREAD_TILE_N; a++) {
                #pragma unroll
                for (int b = 0; b < THREAD_TILE_N; b++) {
                    acc[i][j][a][b] = 0.0;
                }
            }
        }
    }

    const auto ms = x_cols / K;
    for (int m = 0; m < ms; m++) {

        // copy from global memory to shared memory
        __shared__ T Xs[K][M];
        __shared__ T Ws[K][N];
        __syncthreads();
        {
            #pragma unroll
            for (int i0 = 0; i0 < M; i0 += THREADS/K) {
                auto i = threadIdx.x / K;
                auto k = threadIdx.x % K;
                Xs[k][i0+i] = X0(shm_row+i0+i, m*K+k) * G0(shm_row+i0+i, (m*K)/128, col0/g_col);
            }
        }
        {
            auto k0 = threadIdx.x / N * K/2;
            auto idx = threadIdx.x % N;
            #pragma unroll
            for (int k = 0; k < K/2; k++) {
                Ws[k0+k][idx] = W0(m*K+k0+k, shm_col+idx);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K; k++) {
            T xx[2][THREAD_TILE_M];
            T ww[2][THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                auto r_offset = i * 32;
                auto c_offset = i * 16;
                #pragma unroll
                for (int r = 0; r < THREAD_TILE_M; r++) {
                    // * G0(row0+r_offset+r, (m*K)/128, col0/g_col)
                    xx[i][r] = Xs[k][warp_row*128/2 + thread_row*THREAD_TILE_M + r_offset + r];
                }
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    ww[i][c] = Ws[k][warp_col*128/4 + thread_col*THREAD_TILE_N + c_offset + c];
                }
            }

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                auto r_offset = i * 32;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    auto c_offset = j * 16;
                    #pragma unroll
                    for (int r = 0; r < THREAD_TILE_M; r++) {
//                        T gate = G0(row0+r_offset+r, (m*K)/128, (col0+c_offset)/g_col)
//                        T gate = G0(row0+r_offset+r, (m*K)/128, col0/g_col);
                        #pragma unroll
                        for (int c = 0; c < THREAD_TILE_N; c++) {
                            acc[i][j][r][c] += xx[i][r] * ww[j][c];
                        }
                    }
                }
            }
        }
    }

    // write back to global memory
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        auto r_offset = i * 32;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            auto c_offset = j * 16;
            #pragma unroll
            for (int r = 0; r < THREAD_TILE_N; r++) {
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    Y0(row0+r_offset+r,col0+c_offset+c) = acc[i][j][r][c];
                }
            }
        }
    }
}

void fast_gmv_forward(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
    const auto y_rows = Y.size(0);
    const auto y_cols = Y.size(1);
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto w_rows = W.size(0);
    const auto w_cols = W.size(1);
    const auto g_dim1 = G.size(1);
    const auto g_dim2 = G.size(2);

    if (y_rows % 128 != 0) AT_ERROR("y_rows must respect tile size");
    if (y_cols % 128 != 0) AT_ERROR("y_cols must respect tile size");
    if (x_cols % K != 0) AT_ERROR("x_cols must respect tile size");

    const auto g_row = w_rows / g_dim1;
    const auto g_col = w_cols / g_dim2;
    if (g_row % 128 != 0) AT_ERROR("gate block size should respect tile size");
    if (g_col % 128 != 0) AT_ERROR("gate block size should respect tile size");

    dim3 grid(y_cols/N, y_rows/M);
    dim3 block(THREADS);
    if (Y.type().scalarType() == torch::ScalarType::Double) {
        gmv_128x128x8<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), y_rows, y_cols, x_cols, g_dim1, g_dim2);
    } else {
        gmv_128x128x8<<<grid, block>>>(Y.data<float>(), X.data<float>(), W.data<float>(), G.data<float>(), y_rows, y_cols, x_cols, g_dim1, g_dim2);
    }
}

template<typename T> __global__ void gmv_gradw_128x128x8(T *dY, T *X, T *dW, T *G, int w_rows, int w_cols, int x_rows, int g_dim1, int g_dim2) {
    const auto warp = threadIdx.x / WARP_SIZE;          // 8 warps
    const auto warp_row = warp / 4;                     // 2 (different order, does it matter?)
    const auto warp_col = warp % 4;                     // 4
    const auto thread = threadIdx.x % WARP_SIZE;        // 32 threads in a warp
    const auto thread_row = thread / 4;                 // 8 rows in a warp
    const auto thread_col = thread % 4;                 // 4 cols in a warp

    const auto shm_row = blockIdx.y * M;
    const auto shm_col = blockIdx.x * N;
    const auto row0 = shm_row + warp_row*128/2 + thread_row*THREAD_TILE_M;
    const auto col0 = shm_col + warp_col*128/4 + thread_col*THREAD_TILE_N;

    const auto x_cols = w_rows;
    const auto y_cols = w_cols;
//    const auto w_rows = cols;
//    const auto g_row = w_rows / g_dim1;
//    const auto g_col = w_cols / g_dim2;         // 384 / 3 = 128

    T acc[2][2][THREAD_TILE_M][THREAD_TILE_N];

    // initialize 16x16 block of accumulators
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int a = 0; a < THREAD_TILE_N; a++) {
                #pragma unroll
                for (int b = 0; b < THREAD_TILE_N; b++) {
                    acc[i][j][a][b] = 0.0;
                }
            }
        }
    }

    const auto ms = x_rows / K;
    for (int m = 0; m < ms; m++) {
        // copy from global memory to shared memory
        __shared__ T Xs[K][M];
        __shared__ T Ys[K][N];
        __shared__ T Gs[K];
        __syncthreads();
        {
            auto k0 = threadIdx.x / N * K/2;
            auto idx = threadIdx.x % N;
            #pragma unroll
            for (int k = 0; k < K/2; k++) {
                Xs[k0+k][idx] = Xt(shm_row+idx, m*K+k0+k);
            }
            #pragma unroll
            for (int k = 0; k < K/2; k++) {
                Ys[k0+k][idx] = DY(m*K+k0+k, shm_col+idx);
            }
            if (idx < K) {
                Gs[idx] = G0(m*K+idx, row0/128, col0/128);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K; k++) {
            T xx[2][THREAD_TILE_M];
            T yy[2][THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                auto r_offset = i * 32;
                auto c_offset = i * 16;
                #pragma unroll
                for (int r = 0; r < THREAD_TILE_M; r++) {
                    xx[i][r] = Xs[k][warp_row*128/2 + thread_row*THREAD_TILE_M + r_offset + r];
                }
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    yy[i][c] = Ys[k][warp_col*128/4 + thread_col*THREAD_TILE_N + c_offset + c];
                }
            }

            T gate = Gs[k];

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                auto r_offset = i * 32;
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    auto c_offset = j * 16;
                    #pragma unroll
                    for (int r = 0; r < THREAD_TILE_M; r++) {
                        #pragma unroll
                        for (int c = 0; c < THREAD_TILE_N; c++) {
//                            T gate = G0(m*K+k, (row0+r_offset+r)/128, (col0+c_offset+c)/128);
                            acc[i][j][r][c] += xx[i][r] * yy[j][c] * gate;
                        }
                    }
                }
            }
        }
    }

    // write back to global memory
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        auto r_offset = i * 32;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            auto c_offset = j * 16;
            #pragma unroll
            for (int r = 0; r < THREAD_TILE_N; r++) {
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    DW(row0+r_offset+r,col0+c_offset+c) = acc[i][j][r][c];
                }
            }
        }
    }
}

void fast_gmv_gradw(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G) {
    const auto y_rows = dY.size(0);
    const auto y_cols = dY.size(1);
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto w_rows = dW.size(0);
    const auto w_cols = dW.size(1);
    const auto g_dim1 = G.size(1);
    const auto g_dim2 = G.size(2);

    if (y_rows % 128 != 0) AT_ERROR("y_rows must respect tile size");
    if (y_cols % 128 != 0) AT_ERROR("y_cols must respect tile size");
    if (x_cols % K != 0) AT_ERROR("x_cols must respect tile size");

    const auto g_row = w_rows / g_dim1;
    const auto g_col = w_cols / g_dim2;
    if (g_row % 128 != 0) AT_ERROR("gate block size should respect tile size");
    if (g_col % 128 != 0) AT_ERROR("gate block size should respect tile size");

    dim3 grid(w_cols/N, w_rows/M);
    dim3 block(256);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradw_128x128x8<<<grid, block>>>(dY.data<double>(), X.data<double>(), dW.data<double>(), G.data<double>(), w_rows, w_cols, x_rows, g_dim1, g_dim2);
    } else {
        gmv_gradw_128x128x8<<<grid, block>>>(dY.data<float>(), X.data<float>(), dW.data<float>(), G.data<float>(), w_rows, w_cols, x_rows, g_dim1, g_dim2);
    }
}

//#define     NUMBER_OF_THREADS   32
#define     NUMBER_OF_THREADS   128             // WARNING: should be smaller or equal to BLOCK_SIZE
#define     BLOCK_SIZE          128
#define     TILE_SIZE           32              // WARNING: TILE_SIZE cannot be smaller than 32 (warp size)

// TODO: try to make a 256 thread version?
template<typename T> __global__ void gmv_gradg(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * NUMBER_OF_THREADS;
    const auto l = threadIdx.x;
    const auto g2 = blockIdx.y / g_dim1;
    const auto g1 = blockIdx.y % g_dim1;

    T acc = 0;

    __shared__ T Ws[TILE_SIZE][NUMBER_OF_THREADS];
    T Xs[TILE_SIZE];
    T Ys[NUMBER_OF_THREADS];

    #pragma unroll
    for (int a = 0; a < BLOCK_SIZE/TILE_SIZE; a++) {
        #pragma unroll
        for (int b = 0; b < BLOCK_SIZE/NUMBER_OF_THREADS; b++) {
            __syncthreads();
            // parallel loading of weight matrix into shared memory
            #pragma unroll
            for (int u = 0; u < TILE_SIZE; u++) {
                Ws[u][l] = W0(g1*BLOCK_SIZE + a*TILE_SIZE + u, g2*BLOCK_SIZE + b*NUMBER_OF_THREADS + l);
            }

            // load part of X and dY into registers
            #pragma unroll
            for (int u = 0; u < TILE_SIZE; u++) {
                Xs[u] = X0(l0+l, g1*BLOCK_SIZE + a*TILE_SIZE + u);
            }
            #pragma unroll
            for (int v = 0; v < NUMBER_OF_THREADS; v++) {
                Ys[v] = DY(l0+l, g2*BLOCK_SIZE + b*NUMBER_OF_THREADS + v);
            }
            __syncthreads();

            #pragma unroll
            for (int u = 0; u < TILE_SIZE; u++) {
                #pragma unroll
                for (int v = 0; v < NUMBER_OF_THREADS; v++) {
                    acc += Xs[u] * Ws[u][v] * Ys[v];
                }
            }
        }
    }

    DG(l0+l,g1,g2) = acc;
}

void fast_gmv_gradg(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

//    printf("fast_gmv_gradg: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
    if (x_rows % NUMBER_OF_THREADS != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/NUMBER_OF_THREADS, g_dim1*g_dim2);
    dim3 block(NUMBER_OF_THREADS);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradg<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
}


/*
#define     RUNS                4
#define     BLOCK_SIZE          128
#define     ROWS                8
#define     CHUNK_SIZE_1        4
#define     CHUNK_SIZE_2        8
#define     CHUNKS              (CHUNK_SIZE_1*CHUNK_SIZE_2)

template<typename T> __global__ void gmv_gradg_2(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * ROWS;
    const auto g2 = blockIdx.y / g_dim1;
    const auto g1 = blockIdx.y % g_dim1;
    const auto l = threadIdx.x / CHUNKS;
    const auto chunk = threadIdx.x % CHUNKS;
    const auto chunk1 = chunk / CHUNK_SIZE_2;
    const auto chunk2 = chunk % CHUNK_SIZE_2;

    __shared__ T output[CHUNK_SIZE_1][CHUNK_SIZE_2][ROWS];
    int offset_u = g1*BLOCK_SIZE + chunk1*BLOCK_SIZE/CHUNK_SIZE_1;
    int offset_v = g2*BLOCK_SIZE + chunk2*BLOCK_SIZE/CHUNK_SIZE_2;

    __shared__ T Ws[CHUNK_SIZE_1][CHUNK_SIZE_2][BLOCK_SIZE/CHUNK_SIZE_1/RUNS][BLOCK_SIZE/CHUNK_SIZE_2];
    T Xs[BLOCK_SIZE / CHUNK_SIZE_1];
    T Ys[BLOCK_SIZE / CHUNK_SIZE_2];

    __syncthreads();
    #pragma unroll
    for (int u = 0; u < BLOCK_SIZE / CHUNK_SIZE_1; u++) {
        Xs[u] = X0(l0+l, offset_u + u);
    }
    #pragma unroll
    for (int v = 0; v < BLOCK_SIZE / CHUNK_SIZE_2; v++) {
        Ys[v] = DY(l0+l, offset_v + v);
    }
    __syncthreads();

    T acc = 0;

    #pragma unroll
    for (int w = 0; w < RUNS; w++) {
        int offset_run = w * BLOCK_SIZE/CHUNK_SIZE_1/RUNS;          // 8

        __syncthreads();
        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE/CHUNK_SIZE_1/RUNS; u++) {
            #pragma unroll
            for (int v = 0; v < BLOCK_SIZE/CHUNK_SIZE_2; v++) {
                Ws[chunk1][chunk2][u][v] = W0(offset_run + offset_u + u, offset_v + v);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE/CHUNK_SIZE_1/RUNS; u++) {
            #pragma unroll
            for (int v = 0; v < BLOCK_SIZE/CHUNK_SIZE_2; v++) {
                acc += Ws[chunk1][chunk2][u][v] * Xs[offset_run+u] * Ys[v];
            }
        }
    }

    output[chunk1][chunk2][l] = acc;

    __syncthreads();
    if (chunk == 0) {
        T sum = 0;
        #pragma unroll
        for (int a = 0; a < CHUNK_SIZE_1; a++) {
            #pragma unroll
            for (int b = 0; b < CHUNK_SIZE_2; b++) {
                sum += output[a][b][l];
            }
        }
        DG(l0+l,g1,g2) = sum;
    }
    __syncthreads();
}

void fast_gmv_gradg_2(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

    printf("fast_gmv_gradg_2: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
//    if (x_rows % NUMBER_OF_THREADS != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/ROWS, g_dim1*g_dim2);
    dim3 block(ROWS*CHUNK_SIZE_1*CHUNK_SIZE_2);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradg_2<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg_2<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
}
*/

/*
#define     NUMBER_OF_THREADS   64
#define     BLOCK_SIZE          128
#define     TILE_SIZE_X         32
#define     TILE_SIZE_Y         64

template<typename T> __global__ void gmv_gradg_2(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * NUMBER_OF_THREADS;
    const auto l = threadIdx.x;
    const auto g2 = blockIdx.y / g_dim1;
    const auto g1 = blockIdx.y % g_dim1;

    T acc = 0;

    __shared__ T Ws[TILE_SIZE_X][TILE_SIZE_Y];
    T Xs[TILE_SIZE_X];
    T Ys[TILE_SIZE_Y];

    #pragma unroll
    for (int a = 0; a < BLOCK_SIZE/TILE_SIZE_X; a++) {
        #pragma unroll
        for (int b = 0; b < BLOCK_SIZE/TILE_SIZE_Y; b++) {
            __syncthreads();
            // parallel loading of weight matrix into shared memory
//            #pragma unroll
//            for (int u = 0; u < TILE_SIZE_X * TILE_SIZE_Y / NUMBER_OF_THREADS; u++) {
//                const auto u0 = l / TILE_SIZE_Y * TILE_SIZE_X * TILE_SIZE_Y / NUMBER_OF_THREADS;
//                #pragma unroll
//                for (int v = 0; v < TILE_SIZE_Y / NUMBER_OF_THREADS; v++) {
//                    const auto v0 = v * NUMBER_OF_THREADS;
//                    Ws[u0 + u][v0 + l % TILE_SIZE_Y] = W0(g1*BLOCK_SIZE + a*TILE_SIZE_X + u0 + u, g2*BLOCK_SIZE + b*TILE_SIZE_Y + v0 + l % TILE_SIZE_Y);
//                }
//            }
            #pragma unroll
            for (int u = 0; u < TILE_SIZE_X; u++) {
                #pragma unroll
                for (int v0 = 0; v0 < TILE_SIZE_Y; v0 += NUMBER_OF_THREADS) {
                    Ws[u][v0+l] = W0(g1*BLOCK_SIZE + a*TILE_SIZE_X + u, g2*BLOCK_SIZE + b*TILE_SIZE_Y + v0 + l);
                }
            }

            // load part of X and dY into registers
            #pragma unroll
            for (int u = 0; u < TILE_SIZE_X; u++) {
                Xs[u] = X0(l0+l, g1*BLOCK_SIZE + a*TILE_SIZE_X + u);
            }
            #pragma unroll
            for (int v = 0; v < TILE_SIZE_Y; v++) {
                Ys[v] = DY(l0+l, g2*BLOCK_SIZE + b*TILE_SIZE_Y + v);
            }
            __syncthreads();

            #pragma unroll
            for (int u = 0; u < TILE_SIZE_X; u++) {
                #pragma unroll
                for (int v = 0; v < TILE_SIZE_Y; v++) {
                    acc += Xs[u] * Ws[u][v] * Ys[v];
                }
            }
        }
    }

    DG(l0+l,g1,g2) = acc;
}

void fast_gmv_gradg_2(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

//    printf("fast_gmv_gradg: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
    if (x_rows % NUMBER_OF_THREADS != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/NUMBER_OF_THREADS, g_dim1*g_dim2);
    dim3 block(NUMBER_OF_THREADS);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradg_2<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg_2<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
}
*/


#define     WARP_SIZE   128
#define     BLOCK_SIZE  128
#define     GATES_X     2
#define     GATES_Y     2
#define     TILE_X      128
#define     TILE_Y      2

template<typename T> __global__ void gmv_gradg_2(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * WARP_SIZE;
    const auto l = threadIdx.x;
    const auto g2 = (blockIdx.y / (g_dim1/GATES_X)) * GATES_Y;
    const auto g1 = (blockIdx.y % (g_dim1/GATES_X)) * GATES_X;

    T acc[GATES_X][GATES_Y];

    #pragma unroll
    for (int a = 0; a < GATES_X; a++) {
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            acc[a][b] = 0.0;
        }
    }

    __shared__ T Ws[GATES_X][GATES_Y][TILE_X][TILE_Y];
    T Xs[GATES_X][TILE_X];
    T Ys[GATES_Y][TILE_Y];

    #pragma unroll
    for (int u = 0; u < BLOCK_SIZE / TILE_X; u++) {
        #pragma unroll
        for (int a = 0; a < GATES_X; a++) {
            #pragma unroll
            for (int tx = 0; tx < TILE_X; tx++) {
                Xs[a][tx] = X0(l0+l, (g1+a)*BLOCK_SIZE + u*TILE_X + tx);
            }
        }

        #pragma unroll
        for (int v = 0; v < BLOCK_SIZE / TILE_Y; v++) {
            #pragma unroll
            for (int b = 0; b < GATES_Y; b++) {
                #pragma unroll
                for (int ty = 0; ty < TILE_Y; ty++) {
                    Ys[b][ty] = DY(l0+l, (g2+b)*BLOCK_SIZE + v*TILE_Y + ty);
                }
            }

            __syncthreads();
            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int b = 0; b < GATES_Y; b++) {
                    #pragma unroll
                    for (int i = 0; i < TILE_X / WARP_SIZE; i++) {
                        int tx = i * WARP_SIZE + l;
                        #pragma unroll
                        for (int ty = 0; ty < TILE_Y; ty++) {
                            Ws[a][b][tx][ty] = W0((g1+a)*BLOCK_SIZE + u*TILE_X + tx, (g2+b)*BLOCK_SIZE + v*TILE_Y + ty);
                        }
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int b = 0; b < GATES_Y; b++) {
                    #pragma unroll
                    for (int ty = 0; ty < TILE_Y; ty++) {
                        T tmp = 0;
                        #pragma unroll
                        for (int tx = 0; tx < TILE_X; tx++) {
                            tmp += Xs[a][tx] * Ws[a][b][tx][ty];
                        }
                        acc[a][b] += tmp * Ys[b][ty];
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int a = 0; a < GATES_X; a++) {
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            DG(l0+l,g1+a,g2+b) = acc[a][b];
        }
    }
}

void fast_gmv_gradg_2(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

//    printf("fast_gmv_gradg: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
    if (x_rows % WARP_SIZE != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");
    if (g_dim1 % GATES_X != 0) AT_ERROR("gate dim 1 should be multiple of ", GATES_X);
    if (g_dim2 % GATES_Y != 0) AT_ERROR("gate dim 2 should be multiple of ", GATES_Y);

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/WARP_SIZE, g_dim1*g_dim2/GATES_X/GATES_Y);
    dim3 block(WARP_SIZE);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradg_2<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg_2<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
}


//#define     WARP_SIZE   256
//#define     BLOCK_SIZE  128
//#define     GATES_X     2
//#define     GATES_Y     2
//#define     TILE_X      4
//#define     TILE_Y      64

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     1
//#define     GATES_Y     1
//#define     TILE_X      2
//#define     TILE_Y      64
//time: 12.826878309249878

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     4
//#define     GATES_Y     1
//#define     TILE_X      2
//#define     TILE_Y      64
// time: 7.644724130630493

//#define     WARP_SIZE   64
//#define     BLOCK_SIZE  128
//#define     GATES_X     2
//#define     GATES_Y     2
//#define     TILE_X      4
//#define     TILE_Y      32
//time: 7.380401134490967

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     2
//#define     GATES_Y     2
//#define     TILE_X      4
//#define     TILE_Y      32
//time: 7.291672229766846

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     1
//#define     GATES_Y     2
//#define     TILE_X      4
//#define     TILE_Y      64
// time: 6.258315801620483

//#define     WARP_SIZE   256
//#define     BLOCK_SIZE  128
//#define     GATES_X     1
//#define     GATES_Y     2
//#define     TILE_X      4
//#define     TILE_Y      64
// time: 6.180660247802734

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     1
//#define     GATES_Y     4
//#define     TILE_X      2
//#define     TILE_Y      64
// time: 5.30932092666626

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     2
//#define     GATES_Y     4
//#define     TILE_X      2
//#define     TILE_Y      64
// time: 5.3069844245910645

//#define     WARP_SIZE   128
//#define     BLOCK_SIZE  128
//#define     GATES_X     2
//#define     GATES_Y     2
//#define     TILE_X      4
//#define     TILE_Y      64
// time: 4.74586820602417

//#define     WARP_SIZE   64
//#define     BLOCK_SIZE  128
//#define     GATES_X     2
//#define     GATES_Y     2
//#define     TILE_X      1
//#define     TILE_Y      64
//time: 4.216179609298706

#define     WARP_SIZE   128
#define     BLOCK_SIZE  128
#define     GATES_X     2
#define     GATES_Y     2
#define     TILE_X      2
#define     TILE_Y      64
//time: 4.052643060684204




template<typename T> __global__ void gmv_gradg_3(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * WARP_SIZE;
    const auto l = threadIdx.x;
    const auto g2 = (blockIdx.y / (g_dim1/GATES_X)) * GATES_Y;
    const auto g1 = (blockIdx.y % (g_dim1/GATES_X)) * GATES_X;

    T acc[GATES_X][GATES_Y];                                // 2x2 = 4

    #pragma unroll
    for (int a = 0; a < GATES_X; a++) {
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            acc[a][b] = 0.0;
        }
    }

    __shared__ T Ws[GATES_X][GATES_Y][TILE_X][TILE_Y];      // 2x2x4x64 = 4kb
    T Xs[GATES_X][TILE_X];                                  // 2x4 = 8
    T Ys[GATES_Y][TILE_Y];                                  // 2x64 = 128



    #pragma unroll
    for (int v = 0; v < BLOCK_SIZE / TILE_Y; v++) {
        // load in 2x64 floats into local registers
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            #pragma unroll
            for (int ty = 0; ty < TILE_Y; ty++) {
                Ys[b][ty] = DY(l0+l, (g2+b)*BLOCK_SIZE + v*TILE_Y + ty);
            }
        }

        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE / TILE_X; u++) {
            // load in 2x4 floats into local registers
            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int tx = 0; tx < TILE_X; tx++) {
                    Xs[a][tx] = X0(l0+l, (g1+a)*BLOCK_SIZE + u*TILE_X + tx);
                }
            }

            __syncthreads();
            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int b = 0; b < GATES_Y; b++) {
                    #pragma unroll
                    for (int tx0 = 0; tx0 < TILE_X; tx0 += WARP_SIZE/TILE_Y) {
                        int tx = l / TILE_Y + tx0;
                        int ty = l % TILE_Y;
                        Ws[a][b][tx][ty] = W0((g1+a)*BLOCK_SIZE + u*TILE_X + tx, (g2+b)*BLOCK_SIZE + v*TILE_Y + ty);
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int b = 0; b < GATES_Y; b++) {
                    #pragma unroll
                    for (int tx = 0; tx < TILE_X; tx++) {
                        T tmp = 0;
                        #pragma unroll
                        for (int ty = 0; ty < TILE_Y; ty++) {
                            tmp += Ws[a][b][tx][ty] * Ys[b][ty];
                        }
                        acc[a][b] += Xs[a][tx] * tmp;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int a = 0; a < GATES_X; a++) {
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            DG(l0+l,g1+a,g2+b) = acc[a][b];
        }
    }
}

void fast_gmv_gradg_3(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

//    printf("fast_gmv_gradg: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
    if (x_rows % WARP_SIZE != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");
    if (g_dim1 % GATES_X != 0) AT_ERROR("gate dim 1 should be multiple of ", GATES_X);
    if (g_dim2 % GATES_Y != 0) AT_ERROR("gate dim 2 should be multiple of ", GATES_Y);

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/WARP_SIZE, g_dim1*g_dim2/GATES_X/GATES_Y);
    dim3 block(WARP_SIZE);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradg_3<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg_3<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
}

/*
#define     NUMBER_OF_THREADS   256
#define     BLOCK_SIZE          128
#define     TILE_X              4
#define     TILE_Y              64

#define     OPP_X               1
#define     OPP_Y               64         // 16

template<typename T> __global__ void gmv_gradg_4(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * NUMBER_OF_THREADS;
    const auto g2 = blockIdx.y / g_dim1;
    const auto g1 = blockIdx.y % g_dim1;

    __shared__ T Ws[TILE_X][TILE_Y];        // 32x128 (16k), 16x128 (8k), 8x128 (4k)
    T Xs[TILE_X];                           // 16 regs
    T Ys[TILE_Y];                           // 64 regs

    T ww[OPP_Y][OPP_X];                     // 128 regs

    T acc = 0;

    #pragma unroll
    for (int a = 0; a < BLOCK_SIZE; a += BLOCK_SIZE/TILE_X) {
        #pragma unroll
        for (int u = 0; u < TILE_X; u++) {
            Xs[u] = X0(l0 + threadIdx.x, g1*BLOCK_SIZE + a + u);
        }

        #pragma unroll
        for (int b = 0; b < BLOCK_SIZE; b += BLOCK_SIZE/TILE_Y) {
            // load from global memory to shared memory
            __syncthreads();
            #pragma unroll
            for (int i0 = 0; i0 < TILE_X; i0 += NUMBER_OF_THREADS/TILE_Y) {
                int i = threadIdx.x / TILE_Y + i0;
                int j = threadIdx.x % TILE_Y;
                Ws[i][j] = W0(g1*BLOCK_SIZE + a + i, g2*BLOCK_SIZE + b + j);
            }
            __syncthreads();

            #pragma unroll
            for (int v = 0; v < TILE_Y; v++) {
                Ys[v] = DY(l0 + threadIdx.x, g2*BLOCK_SIZE + b + v);
            }

            #pragma unroll
            for (int r = 0; r < TILE_X; r += OPP_X) {
                #pragma unroll
                for (int c = 0; c < TILE_Y; c += OPP_Y) {
                    // loading from shared memory to registers
                    #pragma unroll
                    for (int u = 0; u < OPP_X; u++) {
                        #pragma unroll
                        for (int v = 0; v < OPP_Y; v++) {
                            ww[u][v] = Ws[r+u][c+v];
                        }
                    }

                    #pragma unroll
                    for (int u = 0; u < OPP_X; u++) {
                        T tmp = 0;
                        #pragma unroll
                        for (int v = 0; v < OPP_Y; v++) {
                            tmp += ww[u][v] * Ys[v];
                        }
                        acc += Xs[u] * tmp;
                    }
                }
            }
        }
    }

    DG(l0 + threadIdx.x, g1, g2) = acc;
}

void fast_gmv_gradg_4(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

//    printf("fast_gmv_gradg: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
    if (x_rows % NUMBER_OF_THREADS != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");
    if (g_dim1 % GATES_X != 0) AT_ERROR("gate dim 1 should be multiple of ", GATES_X);
    if (g_dim2 % GATES_Y != 0) AT_ERROR("gate dim 2 should be multiple of ", GATES_Y);

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/NUMBER_OF_THREADS, g_dim1*g_dim2);
    dim3 block(NUMBER_OF_THREADS);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_gradg_4<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg_4<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
}
*/


#define     WARP_SIZE   128
#define     BLOCK_SIZE  128
#define     GATES_X     2
#define     GATES_Y     2
#define     TILE_X      2
#define     TILE_Y      64
#define     NUM         4
//#define     TRANS_Y     32                // time: 3.4938600063323975 (float only)
#define     TRANS_Y     16              // time: 3.5768959522247314
//#define     TRANS_Y     8               // time: 4.367784023284912



template<typename T> __global__ void gmv_gradg_4(T *dY, T *X, T *W, T *dG, int x_cols, int y_cols, int w_cols, int g_dim1, int g_dim2) {
    const auto l0 = blockIdx.x * WARP_SIZE;
    const auto l = threadIdx.x;
    const auto g2 = (blockIdx.y / (g_dim1/GATES_X)) * GATES_Y;
    const auto g1 = (blockIdx.y % (g_dim1/GATES_X)) * GATES_X;

    T acc[GATES_X][GATES_Y];                                // 2x2 = 4

    #pragma unroll
    for (int a = 0; a < GATES_X; a++) {
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            acc[a][b] = 0.0;
        }
    }

    T Xs[GATES_X][TILE_X];                                  // 2x4 = 8
    T Ys[GATES_Y][TILE_Y];                                  // 2x64 = 128

    #pragma unroll
    for (int v = 0; v < BLOCK_SIZE / TILE_Y; v++) {
        // load in 2x64 floats into local registers
//        #pragma unroll
//        for (int b = 0; b < GATES_Y; b++) {
//            #pragma unroll
//            for (int ty = 0; ty < TILE_Y; ty++) {
//                Ys[b][ty] = DY(l0+l, (g2+b)*BLOCK_SIZE + v*TILE_Y + ty);
//            }
//        }
        {
            #pragma unroll
            for (int b = 0; b < GATES_Y; b++) {
                __shared__ T yy[WARP_SIZE][TRANS_Y];
                #pragma unroll
                for (int t = 0; t < TILE_Y; t += TRANS_Y) {
                    __syncthreads();
                    #pragma unroll
                    for (int i0 = 0; i0 < WARP_SIZE; i0 += WARP_SIZE/TRANS_Y) {
                        const auto i = threadIdx.x / TRANS_Y;
                        const auto ty = threadIdx.x % TRANS_Y;
                        yy[i0+i][ty] = DY(l0+i0+i, (g2+b)*BLOCK_SIZE + v*TILE_Y + t+ty);
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int ty = 0; ty < TRANS_Y; ty++) {
                        Ys[b][t+ty] = yy[threadIdx.x][ty];
                    }
                }
            }
        }

        __shared__ T XX[NUM][GATES_X][TILE_X][WARP_SIZE];       // 4x2x2x128    = 8kb

        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE / TILE_X; u++) {
            if (u % NUM == 0) {
                __syncthreads();
//                #pragma unroll
//                for (int uu = 0; uu < 2; uu++) {
//                    #pragma unroll
//                    for (int a = 0; a < GATES_X; a++) {
//                        #pragma unroll
//                        for (int tx = 0; tx < TILE_X; tx++) {
//                            XX[uu][a][tx][l] = X0(l0+l, (g1+a)*BLOCK_SIZE + (u+uu)*TILE_X + tx);
//                        }
//                    }
//                }

                #pragma unroll
                for (int i = 0; i < WARP_SIZE; i += WARP_SIZE/TILE_X/NUM) {
                    #pragma unroll
                    for (int a = 0; a < GATES_X; a++) {
                        const auto lx = (l / (TILE_X*NUM));
                        const auto uu = (l % (TILE_X*NUM)) / TILE_X;
                        const auto tx = (l % (TILE_X*NUM)) % TILE_X;
                        XX[uu][a][tx][lx+i] = X0(l0+lx+i, (g1+a)*BLOCK_SIZE + u*TILE_X + l%(NUM*TILE_X));
                    }
                }
                __syncthreads();
            }

            // load in 2x4 floats into local registers
            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int tx = 0; tx < TILE_X; tx++) {
//                    Xs[a][tx] = X0(l0+l, (g1+a)*BLOCK_SIZE + u*TILE_X + tx);
                    Xs[a][tx] = XX[u%NUM][a][tx][l];
                }
            }

            __shared__ T Ws[GATES_X][GATES_Y][TILE_X][TILE_Y];      // 2x2x4x64 = 4kb

            __syncthreads();
            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int b = 0; b < GATES_Y; b++) {
                    #pragma unroll
                    for (int tx0 = 0; tx0 < TILE_X; tx0 += WARP_SIZE/TILE_Y) {
                        int tx = l / TILE_Y + tx0;
                        int ty = l % TILE_Y;
                        Ws[a][b][tx][ty] = W0((g1+a)*BLOCK_SIZE + u*TILE_X + tx, (g2+b)*BLOCK_SIZE + v*TILE_Y + ty);
                    }
                }
            }
            __syncthreads();

            #pragma unroll
            for (int a = 0; a < GATES_X; a++) {
                #pragma unroll
                for (int b = 0; b < GATES_Y; b++) {
                    #pragma unroll
                    for (int tx = 0; tx < TILE_X; tx++) {
                        T tmp = 0;
                        #pragma unroll
                        for (int ty = 0; ty < TILE_Y; ty++) {
                            tmp += Ws[a][b][tx][ty] * Ys[b][ty];
                        }
                        acc[a][b] += Xs[a][tx] * tmp;
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int a = 0; a < GATES_X; a++) {
        #pragma unroll
        for (int b = 0; b < GATES_Y; b++) {
            DG(l0+l,g1+a,g2+b) = acc[a][b];
        }
    }
}

#define NNN 1024
#define MMM 12

__global__ void hihi(float *tmp) {
    __shared__ float memory[MMM][NNN];
    for (int i = 0; i < MMM; i++) {
        for (int j = 0; j < NNN; j++) {
            memory[i][j] = tmp[0];
        }
    }
}

void fast_gmv_gradg_4(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto y_cols = dY.size(1);
    const auto w_cols = W.size(1);

    const auto g_dim1 = dG.size(1);
    const auto g_dim2 = dG.size(2);

//    printf("fast_gmv_gradg: dg=%ldx%ldx%ld %i\n", dG.size(0), g_dim1, g_dim2, BLOCK_SIZE);
    if (x_rows % WARP_SIZE != 0) AT_ERROR("rows should respect number of threads");
    if (W.size(0) % g_dim1 != 0) AT_ERROR("invalid gate dimension 1");
    if (W.size(1) % g_dim2 != 0) AT_ERROR("invalid gate dimension 2");
    if (g_dim1 * BLOCK_SIZE != W.size(0)) AT_ERROR("invalid block size 1");
    if (g_dim2 * BLOCK_SIZE != W.size(1)) AT_ERROR("invalid block size 2");
    if (g_dim1 % GATES_X != 0) AT_ERROR("gate dim 1 should be multiple of ", GATES_X);
    if (g_dim2 % GATES_Y != 0) AT_ERROR("gate dim 2 should be multiple of ", GATES_Y);

//    printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

    dim3 grid(x_rows/WARP_SIZE, g_dim1*g_dim2/GATES_X/GATES_Y);
    dim3 block(WARP_SIZE);
    if (dY.type().scalarType() == torch::ScalarType::Double) {
//        printf("double disabled\n");
        gmv_gradg_4<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    } else {
        gmv_gradg_4<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), x_cols, y_cols, w_cols, g_dim1, g_dim2);
    }
//    hihi<<<grid, block>>>(dY.data<float>());
}
