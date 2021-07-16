#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
namespace {
template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
  const auto t = tanh(z);
  return 1 - (t * t);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
  return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
  const auto e = exp(z);
  const auto d_relu = z < 0.0 ? 0.0 : 1.0;
  return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
}

} // namespace

__global__ void matmul1(float *Y, float *A, float *X, int rows, int cols, int width) {
   int row = threadIdx.y;
   int col = threadIdx.x;
   float tmp = 0;
   for (int k = 0; k < width; k++) {
     float m = A[row*width + k];
     float n = X[k*cols + col];
     tmp += m * n;
   }
   Y[row*cols+col]=tmp;
}

void cuda_mymm1(torch::Tensor y, torch::Tensor A, torch::Tensor x) {
   const auto rows = y.size(0);
   const auto cols = y.size(1);
   const auto width = A.size(1);

   printf("rows: %ld\n", rows);
   printf("cols: %ld\n", cols);
   printf("width: %ld\n", width);

   dim3 grid(1,1);
   dim3 block(cols, rows);
   matmul1<<<grid, block>>>(y.data<float>(), A.data<float>(), x.data<float>(), rows, cols, width);
}

__global__ void matmul2_8_8(float *Y, float *A, float *X, int rows, int cols, int width) {
  int r_begin = threadIdx.y*8;
  int c_begin = threadIdx.x*8;
  int r_end = r_begin + 8;
  int c_end = c_begin + 8;

  for (int r = r_begin; r < r_end; r++) {
     for (int c = c_begin; c < c_end; c++) {
        int row = r;
        int col = c;
        float tmp = 0;
        for (int k = 0; k < width; k++) {
           float m = A[row*width + k];
           float n = X[k*cols + col];
           tmp += m * n;
        }
        Y[row*cols+col]=tmp;
     }
  }
}

void cuda_mymm2(torch::Tensor y, torch::Tensor A, torch::Tensor x) {
   const auto rows = y.size(0);
   const auto cols = y.size(1);
   const auto width = A.size(1);

   printf("rows: %ld -> %ld\n", rows, rows/8);
   printf("cols: %ld -> %ld\n", cols, cols/8);
   printf("width: %ld\n", width);

   dim3 grid(1,1);
   dim3 block(cols/8, rows/8);
   matmul2_8_8<<<grid, block>>>(y.data<float>(), A.data<float>(), x.data<float>(), rows, cols, width);
}

__global__ void matmul3_8_8(float *Y, float *A, float *X, int rows, int cols, int width) {
  int r_begin = (blockDim.y * blockIdx.y + threadIdx.y)*8;
  int c_begin = (blockDim.x * blockIdx.x + threadIdx.x)*8;
  int r_end = r_begin + 8;
  int c_end = c_begin + 8;

  for (int r = r_begin; r < r_end; r++) {
     for (int c = c_begin; c < c_end; c++) {
        int row = r;
        int col = c;
        float tmp = 0;
        for (int k = 0; k < width; k++) {
           float m = A[row*width + k];
           float n = X[k*cols + col];
           tmp += m * n;
        }
        Y[row*cols+col]=tmp;
     }
  }
}

void cuda_mymm3(torch::Tensor y, torch::Tensor A, torch::Tensor x) {
   const auto rows = y.size(0);
   const auto cols = y.size(1);
   const auto width = A.size(1);

   printf("cuda_mymm3\n");
   printf("rows: %ld -> %ld\n", rows, rows/32/8);
   printf("cols: %ld -> %ld\n", cols, cols/32/8);
   printf("width: %ld\n", width);

   dim3 grid(cols/32/8, rows/32/8);
   dim3 block(32, 32);
   matmul3_8_8<<<grid, block>>>(y.data<float>(), A.data<float>(), x.data<float>(), rows, cols, width);
//   cudaDeviceSynchronize();
}


*/

#define BLOCK_SIZE 32

__global__ void gmm1(float *Y, float *A, float *B, float *G, int rows, int cols, int width, int g_cols) {
   const auto r = threadIdx.y;
   const auto c = threadIdx.x;
   const auto row = blockDim.y * blockIdx.y + r;
   const auto col = blockDim.x * blockIdx.x + c;

   float tmp1 = 0;

   for (int m = 0; m < width / BLOCK_SIZE; m++) {
      const auto offset = m * BLOCK_SIZE;

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
      As[r][c] = A[row*width+c+offset];
      Bs[r][c] = B[(r+offset)*cols+col];

      __syncthreads();
      float tmp2 = 0;
      for (int k = 0; k < BLOCK_SIZE; k++) {
         tmp2 += As[r][k] * Bs[k][c];
      }
      tmp1 += tmp2 * G[m*g_cols+blockIdx.x];
      __syncthreads();
   }

   Y[row*cols+col] = tmp1;
}

void cuda_gmm1(torch::Tensor Y, torch::Tensor A, torch::Tensor B, torch::Tensor G) {
   const auto rows = Y.size(0);
   const auto cols = Y.size(1);
   const auto width = A.size(1);

   const auto g_cols = G.size(1);

   printf("cuda_gmm1: rows=%ld cols=%ld width=%ld stride=%ld blocks=%ld x %ld\n", rows, cols, width, g_cols, cols/BLOCK_SIZE, rows/BLOCK_SIZE);

   dim3 grid(cols/BLOCK_SIZE, rows/BLOCK_SIZE);
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   gmm1<<<grid, block>>>(Y.data<float>(), A.data<float>(), B.data<float>(), G.data<float>(), rows, cols, width, g_cols);
}



__global__ void gmm1_gradg_1(float *gradG, float *X, float *W, float *gradY, int rows, int cols, int width, int g_cols) {
   for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
         for (int w = 0; w < width; w++) {
            gradG[w/BLOCK_SIZE*g_cols+c/BLOCK_SIZE] += X[r*width+w] * W[w*cols+c] * gradY[r*rows+c];
         }
      }
   }
}

__global__ void gmm1_gradg_2(float *gradG, float *X, float *W, float *gradY, int rows, int cols, int width, int g_cols) {
   for (int c = 0; c < cols; c++) {
      for (int w = 0; w < width; w++) {
         float tmp = 0.0;
         for (int r = 0; r < rows; r++) {
            tmp += X[r*width+w] * W[w*cols+c] * gradY[r*rows+c];
         }
         gradG[w/BLOCK_SIZE*g_cols+c/BLOCK_SIZE] += tmp;
      }
   }
}

__global__ void gmm1_gradg_3(float *gradG, float *X, float *W, float *gradY, int rows, int cols, int width, int g_cols) {
   int nc = cols / BLOCK_SIZE;
   int nw = width / BLOCK_SIZE;

   for (int cc = 0; cc < nc; cc++) {
      for (int ww = 0; ww < nw; ww++) {
         int c0 = cc * BLOCK_SIZE;
         int w0 = ww * BLOCK_SIZE;
         float tmp = 0;
         for (int c = 0; c < BLOCK_SIZE; c++) {
            for (int w = 0; w < BLOCK_SIZE; w++) {
               for (int r = 0; r < rows; r++) {
                  tmp += X[r*width+w0+w] * W[(w0+w)*cols+c+c0] * gradY[r*rows+c+c0];
               }
            }
         }
         gradG[ww*g_cols+cc] += tmp;
      }
   }
}

// need grad2: gate:torch.Size([2, 4]) input:torch.Size([256, 64]) weight:torch.Size([64, 128]) doutput:torch.Size([256, 128])
// cuda_gmm1_backward_g: rows=256 cols=64 width=128


void cuda_gmm1_backward_g(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   const auto rows = X.size(0);
   const auto cols = W.size(1);
   const auto width = X.size(1);
   const auto g_rows = gradG.size(0);
   const auto g_cols = gradG.size(1);

   printf("cuda_gmm1_backward_g: rows=%ld cols=%ld width=%ld\n", rows, cols, width);
   printf("blocks: %ld x %ld\n", g_rows, g_cols);
   dim3 grid(1, 1);
   dim3 block(1, 1);
   gmm1_gradg_3<<<grid, block>>>(gradG.data<float>(), X.data<float>(), W.data<float>(), gradY.data<float>(), rows, cols, width, g_cols);
}

__global__ void gmm1_gradg_4(float *gradG, float *X, float *W, float *gradY, int rows, int cols, int width, int g_cols) {
   int c0 = blockIdx.x * BLOCK_SIZE;
   int w0 = blockIdx.y * BLOCK_SIZE;
   float tmp = 0;
   for (int c = 0; c < BLOCK_SIZE; c++) {
      for (int w = 0; w < BLOCK_SIZE; w++) {
         for (int r = 0; r < rows; r++) {
            tmp += X[r*width+w0+w] * W[(w0+w)*cols+c+c0] * gradY[r*rows+c+c0];
         }
      }
   }
   gradG[blockIdx.y*g_cols+blockIdx.x] += tmp;
}

// HERE

void cuda_gmm2_backward_g(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   const auto rows = X.size(0);
   const auto cols = W.size(1);
   const auto width = X.size(1);
   const auto g_rows = gradG.size(0);
   const auto g_cols = gradG.size(1);

   printf("cuda_gmm1_backward_g: rows=%ld cols=%ld width=%ld\n", rows, cols, width);
   printf("blocks: %ld x %ld\n", g_rows, g_cols);
   dim3 grid(cols / BLOCK_SIZE, width / BLOCK_SIZE);
   dim3 block(1, 1);
   gmm1_gradg_4<<<grid, block>>>(gradG.data<float>(), X.data<float>(), W.data<float>(), gradY.data<float>(), rows, cols, width, g_cols);
}


__global__ void gmm1_gradw(float *Y, float *A, float *B, float *G, int rows, int cols, int width, int g_cols) {
   const auto r = threadIdx.y;
   const auto c = threadIdx.x;
   const auto row = blockDim.y * blockIdx.y + r;
   const auto col = blockDim.x * blockIdx.x + c;

   float tmp1 = 0;

   for (int m = 0; m < width / BLOCK_SIZE; m++) {
      const auto offset = m * BLOCK_SIZE;

      __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
      As[r][c] = A[row*width+c+offset];
      Bs[r][c] = B[(r+offset)*cols+col];

      __syncthreads();
      for (int k = 0; k < BLOCK_SIZE; k++) {
         tmp1 += As[r][k] * Bs[k][c];
      }
      __syncthreads();
   }

   const auto b_row = row / BLOCK_SIZE;
   const auto c_row = col / BLOCK_SIZE;
   Y[row*cols+col] = tmp1 * G[b_row*g_cols + c_row];
}

void cuda_gmm1_backward_w(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
   const auto rows = gradW.size(0);
   const auto cols = gradW.size(1);
   const auto width = X.size(1);

   const auto g_cols = G.size(1);

   printf("cuda_gmm1_backward_w: rows=%ld cols=%ld width=%ld\n", rows, cols, width);
   printf("gate-stride: %ld\n", g_cols);
   printf("blocks: %ld x %ld\n", cols/BLOCK_SIZE, rows/BLOCK_SIZE);

   dim3 grid(cols/BLOCK_SIZE, rows/BLOCK_SIZE);
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   gmm1_gradw<<<grid, block>>>(gradW.data<float>(), X.data<float>(), gradY.data<float>(), G.data<float>(), rows, cols, width, g_cols);
}
