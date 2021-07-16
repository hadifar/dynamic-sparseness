#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE  32



__global__ void matmul1(float *Y, float *X, float *W, int rows, int cols, int x_cols, unsigned long long *time) {
    unsigned long long start = clock();
    const auto r = threadIdx.y;
    const auto c = threadIdx.x;
    const auto row = blockDim.y * blockIdx.y + r;
    const auto col = blockDim.x * blockIdx.x + c;

    float tmp = 0;

    for (int k = 0; k < x_cols; k++) {
        tmp += X[row*x_cols+k] * W[k*cols+col];
    }
    Y[row*cols+col] = tmp;

    *time = (clock()-start);
}

void cuda_mm1(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
    const auto y_rows = Y.size(0);
    const auto y_cols = Y.size(1);
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto w_rows = W.size(0);
    const auto w_cols = W.size(1);

    printf("cuda_mm1\n");
    printf("y_rows: %ld   y_cols: %ld\n", y_rows, y_cols);
    printf("x_rows: %ld   x_cols: %ld\n", x_rows, x_cols);
    printf("w_rows: %ld   w_cols: %ld\n", w_rows, w_cols);

    if (y_rows % BLOCK_SIZE != 0) AT_ERROR("y_rows must respect tile size");
    if (y_cols % BLOCK_SIZE != 0) AT_ERROR("y_cols must respect tile size");

   unsigned long long *time;
   cudaMalloc(&time, sizeof(unsigned long long));

   dim3 grid(y_cols/BLOCK_SIZE, y_rows/BLOCK_SIZE);
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   matmul1<<<grid, block>>>(Y.data<float>(), X.data<float>(), W.data<float>(), y_rows, y_cols, x_cols, time);

   unsigned long long value;
   cudaMemcpy(&value, time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
   printf("clock: %llu\n", value);
}



__global__ void matmul4(float *Y, float *X, float *W, int rows, int cols, int x_cols, unsigned long long *time) {
   unsigned long long start = clock();
   const auto r = threadIdx.y;
   const auto c = threadIdx.x;
   const auto row = blockDim.y * blockIdx.y + r;
   const auto col = blockDim.x * blockIdx.x + c;

   float tmp = 0;

   for (int m = 0; m < x_cols / BLOCK_SIZE; m++) {
      const auto offset = m * BLOCK_SIZE;

      __shared__ float Xs[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Ws[BLOCK_SIZE][BLOCK_SIZE];
      Xs[r][c] = X[row*x_cols+c+offset];
      Ws[r][c] = W[(r+offset)*cols+col];

      __syncthreads();
      for (int k = 0; k < BLOCK_SIZE; k++) {
         tmp += Xs[r][k] * Ws[k][c];
      }
      __syncthreads();
   }

   Y[row*cols+col] = tmp;

   *time = (clock()-start);
}

void cuda_mm4(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
    const auto y_rows = Y.size(0);
    const auto y_cols = Y.size(1);
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto w_rows = W.size(0);
    const auto w_cols = W.size(1);

    printf("cuda_mm4\n");
    printf("y_rows: %ld   y_cols: %ld\n", y_rows, y_cols);
    printf("x_rows: %ld   x_cols: %ld\n", x_rows, x_cols);
    printf("w_rows: %ld   w_cols: %ld\n", w_rows, w_cols);

    if (y_rows % BLOCK_SIZE != 0) AT_ERROR("y_rows must respect tile size");
    if (y_cols % BLOCK_SIZE != 0) AT_ERROR("y_cols must respect tile size");

   unsigned long long *time;
   cudaMalloc(&time, sizeof(unsigned long long));

   dim3 grid(y_cols/BLOCK_SIZE, y_rows/BLOCK_SIZE);
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   matmul4<<<grid, block>>>(Y.data<float>(), X.data<float>(), W.data<float>(), y_rows, y_cols, x_cols, time);

   unsigned long long value;
   cudaMemcpy(&value, time, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
   printf("clock: %llu\n", value);
}

//#define     WARP_SIZE       32
//#define     M               128
//#define     N               128
//#define     K               8
//#define     WARP_WIDTH      4
//#define     WARP_HEIGHT     2
//#define     THREADS         256
//#define     THREADS_WIDTH   4
//#define     THREADS_HEIGHT  8
//#define     THREAD_TILE_M   4
//#define     THREAD_TILE_N   4

// does not work yet
#define     WARP_SIZE       32
#define     M               128
#define     N               128
#define     K               8
#define     WARP_WIDTH      4
#define     WARP_HEIGHT     2
#define     THREADS         256
#define     THREADS_WIDTH   4
#define     THREADS_HEIGHT  8
#define     THREAD_TILE_M   4
#define     THREAD_TILE_N   4

#define     Y0(a, b)        Y[(a)*y_cols + (b)]
#define     X0(a, b)        X[(a)*x_cols + (b)]
#define     W0(a, b)        W[(a)*w_cols + (b)]

template<typename T> __global__ void my_sgemm_128x128x8_NN(T *Y, T *X, T *W, int rows, int cols, int x_cols) {
    const auto y_cols = cols;
    const auto w_cols = cols;
    const auto warp = threadIdx.x / WARP_SIZE;          // 8 warps
    const auto warp_row = warp / WARP_WIDTH;            // 2 (different order, does it matter?)
    const auto warp_col = warp % WARP_WIDTH;            // 4
    const auto thread = threadIdx.x % WARP_SIZE;        // 32 threads in a warp
    const auto thread_row = thread / THREADS_WIDTH;     // 8 rows in a warp
    const auto thread_col = thread % THREADS_WIDTH;     // 4 cols in a warp

    const auto shm_row = blockIdx.y * M;
    const auto shm_col = blockIdx.x * N;
    const auto row0 = shm_row + warp_row*M/WARP_HEIGHT + thread_row*THREAD_TILE_M;
    const auto col0 = shm_col + warp_col*N/WARP_WIDTH + thread_col*THREAD_TILE_N;

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
                Xs[k][i0+i] = X0(shm_row+i0+i, m*K+k);
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
                auto r_offset = i*THREADS_HEIGHT*THREAD_TILE_M;
                auto c_offset = i*THREADS_WIDTH*THREAD_TILE_N;
                #pragma unroll
                for (int r = 0; r < THREAD_TILE_M; r++) {
                    xx[i][r] = Xs[k][warp_row*M/WARP_HEIGHT + thread_row*THREAD_TILE_M + r_offset + r];
                }
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    ww[i][c] = Ws[k][warp_col*N/WARP_WIDTH + thread_col*THREAD_TILE_N + c_offset + c];
                }
            }

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int r = 0; r < THREAD_TILE_M; r++) {
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
        auto r_offset = i*THREADS_HEIGHT*THREAD_TILE_M;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            auto c_offset = j*THREADS_WIDTH*THREAD_TILE_N;
            #pragma unroll
            for (int a = 0; a < THREAD_TILE_N; a++) {
                #pragma unroll
                for (int b = 0; b < THREAD_TILE_N; b++) {
                    Y0(row0+r_offset+a,col0+c_offset+b) = acc[i][j][a][b];
                }
            }
        }
    }
}

void cuda_mm5(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
    const auto y_rows = Y.size(0);
    const auto y_cols = Y.size(1);
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto w_rows = W.size(0);
    const auto w_cols = W.size(1);

//    printf("cuda_mm5\n");
//    printf("y_rows: %ld   y_cols: %ld\n", y_rows, y_cols);
//    printf("x_rows: %ld   x_cols: %ld\n", x_rows, x_cols);
//    printf("w_rows: %ld   w_cols: %ld\n", w_rows, w_cols);

    if (y_rows % M != 0) AT_ERROR("y_rows must respect tile size of ", M);
    if (y_cols % N != 0) AT_ERROR("y_cols must respect tile size of ", N);
    if (x_cols % K != 0) AT_ERROR("x_cols must respect tile size of ", K);

    dim3 grid(y_cols/N, y_rows/M);
    dim3 block(THREADS);
    if (Y.type().scalarType() == torch::ScalarType::Double) {
        my_sgemm_128x128x8_NN<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), y_rows, y_cols, x_cols);
    } else {
        my_sgemm_128x128x8_NN<<<grid, block>>>(Y.data<float>(), X.data<float>(), W.data<float>(), y_rows, y_cols, x_cols);
    }
}

#define     Xt(a, b)        X[a + (b)*x_cols]

template<typename T> __global__ void my_sgemm_128x128x8_TN(T *Y, T *X, T *W, int rows, int cols, int x_rows) {
    const auto x_cols = rows;
    const auto y_cols = cols;
    const auto w_cols = cols;
    const auto warp = threadIdx.x / WARP_SIZE;          // 8 warps
    const auto warp_row = warp / WARP_WIDTH;            // 2 (different order, does it matter?)
    const auto warp_col = warp % WARP_WIDTH;            // 4
    const auto thread = threadIdx.x % WARP_SIZE;        // 32 threads in a warp
    const auto thread_row = thread / THREADS_WIDTH;     // 8 rows in a warp
    const auto thread_col = thread % THREADS_WIDTH;     // 4 cols in a warp

    const auto shm_row = blockIdx.y * M;
    const auto shm_col = blockIdx.x * N;
    const auto row0 = shm_row + warp_row*M/WARP_HEIGHT + thread_row*THREAD_TILE_M;
    const auto col0 = shm_col + warp_col*N/WARP_WIDTH + thread_col*THREAD_TILE_N;

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
        __shared__ T Xs[M][K+1];                // +1 reduces bank conflicts
        __shared__ T Ws[K][N];
        __syncthreads();
        auto k0 = threadIdx.x / N * K/2;
        auto idx = threadIdx.x % N;
        #pragma unroll
        for (int k = 0; k < K/2; k++) {
            Xs[idx][k0+k] = Xt(shm_row+idx, m*K+k0+k);
        }
        #pragma unroll
        for (int k = 0; k < K/2; k++) {
            Ws[k0+k][idx] = W0(m*K+k0+k, shm_col+idx);
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K; k++) {
            T xx[2][THREAD_TILE_M];
            T ww[2][THREAD_TILE_N];

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                auto r_offset = i*THREADS_HEIGHT*THREAD_TILE_M;
                auto c_offset = i*THREADS_WIDTH*THREAD_TILE_N;
                #pragma unroll
                for (int r = 0; r < THREAD_TILE_M; r++) {
                    xx[i][r] = Xs[warp_row*M/WARP_HEIGHT + thread_row*THREAD_TILE_M + r_offset + r][k];
                }
                #pragma unroll
                for (int c = 0; c < THREAD_TILE_N; c++) {
                    ww[i][c] = Ws[k][warp_col*N/WARP_WIDTH + thread_col*THREAD_TILE_N + c_offset + c];
                }
            }

            #pragma unroll
            for (int i = 0; i < 2; i++) {
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    #pragma unroll
                    for (int r = 0; r < THREAD_TILE_M; r++) {
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
        auto r_offset = i*THREADS_HEIGHT*THREAD_TILE_M;
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            auto c_offset = j*THREADS_WIDTH*THREAD_TILE_N;
            #pragma unroll
            for (int a = 0; a < THREAD_TILE_N; a++) {
                #pragma unroll
                for (int b = 0; b < THREAD_TILE_N; b++) {
                    Y0(row0+r_offset+a,col0+c_offset+b) = acc[i][j][a][b];
                }
            }
        }
    }
}

void cuda_mm5_TN(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
    const auto y_rows = Y.size(0);
    const auto y_cols = Y.size(1);
    const auto x_rows = X.size(0);
    const auto x_cols = X.size(1);
    const auto w_rows = W.size(0);
    const auto w_cols = W.size(1);

//    printf("cuda_mm5_TN\n");
//    printf("y_rows: %ld   y_cols: %ld\n", y_rows, y_cols);
//    printf("x_rows: %ld   x_cols: %ld\n", x_rows, x_cols);
//    printf("w_rows: %ld   w_cols: %ld\n", w_rows, w_cols);

    if (y_rows % M != 0) AT_ERROR("y_rows must respect tile size of ", M);
    if (y_cols % N != 0) AT_ERROR("y_cols must respect tile size of ", N);
    if (x_rows % K != 0) AT_ERROR("x_rows must respect tile size of ", K);

    dim3 grid(y_cols/N, y_rows/M);
    dim3 block(THREADS);
    if (Y.type().scalarType() == torch::ScalarType::Double) {
        my_sgemm_128x128x8_TN<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), y_rows, y_cols, x_rows);
    } else {
        my_sgemm_128x128x8_TN<<<grid, block>>>(Y.data<float>(), X.data<float>(), W.data<float>(), y_rows, y_cols, x_rows);
    }
}
