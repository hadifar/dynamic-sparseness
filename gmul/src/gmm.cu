#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define BLOCK_SIZE 4



#define X0(a,b,c) X[(a)*x_rows*x_cols+(b)*x_cols+(c)]
#define G0(a,b,c) G[(a)*g_rows*g_cols+(b)*g_cols+(c)]
#define Y0(a,b,c) Y[(a)*y_rows*y_cols+(b)*y_cols+(c)]
#define W0(a,b) W[(a)*w_cols+(b)]

__global__ void gmm_double1(double *Y, double *X, double *W, double *G, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   for (auto b = 0; b < batches; b++) {
     for (auto r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
           double tmp = 0;
           for (int k = 0; k < width; k++) {
              tmp += X0(b,r,k) * W0(k,c) * G0(b,k/dimx,c/dimy);
           }
           Y0(b,r,c) = tmp;
        }
     }
   }
}

void cuda_gmm_forward1(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = Y.size(0);
   const auto rows = Y.size(1);
   const auto cols = Y.size(2);
   const auto width = W.size(0);

   const auto dimx = W.size(0) / G.size(1);
   const auto dimy = W.size(1) / G.size(2);

   printf("cuda_gmm_forward1 -> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(1, 1);
   dim3 block(1, 1);
   gmm_double1<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), batches, rows, cols, width, dimx, dimy);
}


/* more threads active */

__global__ void gmm_double2(double *Y, double *X, double *W, double *G, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto b = blockIdx.x / rows;
   const auto r = blockIdx.x % rows;
   const auto c = blockIdx.y;

   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   double tmp = 0;
   for (int k = 0; k < width; k++) {
      tmp += X0(b,r,k) * W0(k,c) * G0(b,k/dimx,c/dimy);
   }
   Y0(b,r,c) = tmp;
}

void cuda_gmm_forward2(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = Y.size(0);
   const auto rows = Y.size(1);
   const auto cols = Y.size(2);
   const auto width = W.size(0);

   const auto dimx = W.size(0) / G.size(1);
   const auto dimy = W.size(1) / G.size(2);

   printf("cuda_gmm_forward2 -> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(batches*rows, cols);
   dim3 block(1, 1);
   gmm_double2<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), batches, rows, cols, width, dimx, dimy);
}

/* shared memory version */

__global__ void gmm_double3(double *Y, double *X, double *W, double *G, int batches, int rows, int cols, int width, int dimx, int dimy) {
//   const auto b = (blockIdx.x * blockDim.x + threadIdx.x) / rows;
//   const auto r = (blockIdx.x * blockDim.x + threadIdx.x) % rows;
    const auto b = (blockIdx.x * blockDim.x) / rows;
    const auto r0 = (blockIdx.x * blockDim.x) % rows;
    const auto r = threadIdx.x;
    const auto c0 = blockIdx.y * blockDim.y;
    const auto c = threadIdx.y;

    const auto x_rows = rows, x_cols = width;
    const auto g_rows = width/dimx, g_cols = cols/dimy;
    const auto y_rows = rows, y_cols = cols;
    const auto w_cols = cols;

// This works
//   double tmp = 0;
//   int k = 0;
//   for (int m = 0; m < width / BLOCK_SIZE; m++) {
//       for (int i = 0; i < BLOCK_SIZE; i++) {
//          tmp += X0(b,r,k) * W0(k,c) * G0(b,k/dimx,c/dimy);
//          k++;
//       }
//   }
//   Y0(b,r,c) = tmp;

//    double tmp = 0;
//    int k0 = 0;
//    for (int m = 0; m < width / BLOCK_SIZE; m++) {
//        for (int k = 0; k < BLOCK_SIZE; k++) {
//            tmp += X0(b,r,k0+k) * W0(k0+k,c0+c) * G0(b,(k0+k)/dimx,(c0+c)/dimy);
//        }
//        k0 += BLOCK_SIZE;
//    }
//    Y0(b,r,c0+c) = tmp;

    double tmp = 0;
    int k0 = 0;
    for (int m = 0; m < width / BLOCK_SIZE; m++) {
        __shared__ double Xs[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Ws[BLOCK_SIZE][BLOCK_SIZE];
        Xs[threadIdx.x][threadIdx.y] = X0(b,r0+threadIdx.x,k0+threadIdx.y);
        Ws[threadIdx.x][threadIdx.y] = W0(k0+threadIdx.x,c0+threadIdx.y);

        __syncthreads();
        double tmp1 = 0;
        for (int k = 0; k < BLOCK_SIZE; k++) {
            tmp1 += Xs[r][k] * Ws[k][c];
        }
        tmp += tmp1 * G0(b,k0/dimx,c0/dimy);
        k0 += BLOCK_SIZE;
        __syncthreads();
    }
    Y0(b,r0+r,c0+c) = tmp;
}

void cuda_gmm_forward3(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = Y.size(0);
   const auto rows = Y.size(1);
   const auto cols = Y.size(2);
   const auto width = W.size(0);

   const auto dimx = W.size(0) / G.size(1);
   const auto dimy = W.size(1) / G.size(2);

   printf("cuda_gmm_forward3 -> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   if ((batches*rows) % BLOCK_SIZE != 0) AT_ERROR("blocksize error");
   if (cols % BLOCK_SIZE != 0) AT_ERROR("blocksize error");
   if (width % BLOCK_SIZE != 0) AT_ERROR("blocksize error");

   dim3 grid(batches*rows / BLOCK_SIZE, cols / BLOCK_SIZE);
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   gmm_double3<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), batches, rows, cols, width, dimx, dimy);
}


#define X0(a,b,c) X[(a)*x_rows*x_cols+(b)*x_cols+(c)]
#define G0(a,b,c) G[(a)*g_rows*g_cols+(b)*g_cols+(c)]
#define dY(a,b,c) gradY[(a)*y_rows*y_cols+(b)*y_cols+(c)]
#define dW(a,b) gradW[(a)*w_cols+(b)]

__global__ void gmm_gradw_double1(double *gradW, double *X, double *gradY, double *G, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   for (int c = 0; c < cols; c++) {
      for (int k = 0; k < width; k++) {
         double tmp = 0;
         for (auto b = 0; b < batches; b++) {
           for (auto r = 0; r < rows; r++) {
              tmp += X0(b,r,k) * G0(b,k/dimx,c/dimy) * dY(b,r,c);
           }
         }
         dW(k,c) += tmp;
      }
   }
}

void cuda_gmm_backward_w1(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != gradW.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != gradW.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = gradY.size(0);
   const auto rows = gradY.size(1);
   const auto cols = gradY.size(2);
   const auto width = gradW.size(0);

   const auto dimx = gradW.size(0) / G.size(1);
   const auto dimy = gradW.size(1) / G.size(2);

   if (dimx != BLOCK_SIZE) return;
   if (dimy != BLOCK_SIZE) return;

   printf("-> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(1, 1);
   dim3 block(1, 1);
   gmm_gradw_double1<<<grid, block>>>(gradW.data<double>(), X.data<double>(), gradY.data<double>(), G.data<double>(), batches, rows, cols, width, dimx, dimy);
}

// more threads

__global__ void gmm_gradw_double2(double *gradW, double *X, double *gradY, double *G, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto c = blockIdx.x;
   const auto k = blockIdx.y;
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   double tmp = 0;
   for (auto b = 0; b < batches; b++) {
      for (auto r = 0; r < rows; r++) {
         tmp += X0(b,r,k) * G0(b,k/dimx,c/dimy) * dY(b,r,c);
      }
   }
   dW(k,c) += tmp;
}

void cuda_gmm_backward_w2(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != gradW.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != gradW.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = gradY.size(0);
   const auto rows = gradY.size(1);
   const auto cols = gradY.size(2);
   const auto width = gradW.size(0);

   const auto dimx = gradW.size(0) / G.size(1);
   const auto dimy = gradW.size(1) / G.size(2);

   if (dimx != BLOCK_SIZE) return;
   if (dimy != BLOCK_SIZE) return;

   printf("-> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(cols, width);
   dim3 block(1, 1);
   gmm_gradw_double2<<<grid, block>>>(gradW.data<double>(), X.data<double>(), gradY.data<double>(), G.data<double>(), batches, rows, cols, width, dimx, dimy);
}


__global__ void gmm_gradw_double3(double *gradW, double *X, double *gradY, double *G, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto c = blockIdx.x * blockDim.x + threadIdx.x;
   const auto k = blockIdx.y * blockDim.y + threadIdx.y;
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   double tmp = 0;
   for (auto b = 0; b < batches; b++) {
      for (auto r = 0; r < rows; r++) {
         tmp += X0(b,r,k) * G0(b,k/dimx,c/dimy) * dY(b,r,c);
      }
   }
   dW(k,c) += tmp;
}

void cuda_gmm_backward_w3(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != gradW.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != gradW.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = gradY.size(0);
   const auto rows = gradY.size(1);
   const auto cols = gradY.size(2);
   const auto width = gradW.size(0);

   const auto dimx = gradW.size(0) / G.size(1);
   const auto dimy = gradW.size(1) / G.size(2);

   if (dimx != BLOCK_SIZE) return;
   if (dimy != BLOCK_SIZE) return;

   printf("cuda_gmm_backward_w3 -> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(cols / BLOCK_SIZE, width / BLOCK_SIZE);
   dim3 block(BLOCK_SIZE, BLOCK_SIZE);
   gmm_gradw_double3<<<grid, block>>>(gradW.data<double>(), X.data<double>(), gradY.data<double>(), G.data<double>(), batches, rows, cols, width, dimx, dimy);
}



#define X0(a,b,c) X[(a)*x_rows*x_cols+(b)*x_cols+(c)]
#define dG(a,b,c) gradG[(a)*g_rows*g_cols+(b)*g_cols+(c)]
#define dY(a,b,c) gradY[(a)*y_rows*y_cols+(b)*y_cols+(c)]
#define W0(a,b) W[(a)*w_cols+(b)]

__global__ void gmm_gradg_double1(double *gradG, double *X, double *W, double *gradY, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   for (auto b = 0; b < batches; b++) {
     for (auto r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
           for (int k = 0; k < width; k++) {
              dG(b,k/dimx,c/dimy) += X0(b,r,k) * W0(k,c) * dY(b,r,c);
           }
        }
     }
   }

}

void cuda_gmm_backward_g1(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != gradG.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = gradY.size(0);
   const auto rows = gradY.size(1);
   const auto cols = gradY.size(2);
   const auto width = W.size(0);

   const auto dimx = W.size(0) / gradG.size(1);
   const auto dimy = W.size(1) / gradG.size(2);

   printf("-> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(1, 1);
   dim3 block(1, 1);
   gmm_gradg_double1<<<grid, block>>>(gradG.data<double>(), X.data<double>(), W.data<double>(), gradY.data<double>(), batches, rows, cols, width, dimx, dimy);
}

__global__ void gmm_gradg_double2(double *gradG, double *X, double *W, double *gradY, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   const auto b = blockIdx.x / g_rows;
   const auto gr = blockIdx.x % g_rows;
   const auto gc = blockIdx.y;

   double tmp = 0;
   for (int i = 0; i < dimx; i++) {
      for (int j = 0; j < dimy; j++) {
         int k = gr * dimx + i;
         int c = gc * dimy + j;
         for (auto r = 0; r < rows; r++) {
            tmp += X0(b,r,k) * W0(k,c) * dY(b,r,c);
         }
      }
   }
   dG(b,gr,gc) = tmp;
}

void cuda_gmm_backward_g2(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != gradG.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = gradY.size(0);
   const auto rows = gradY.size(1);
   const auto cols = gradY.size(2);
   const auto width = W.size(0);

   const auto dimx = W.size(0) / gradG.size(1);
   const auto dimy = W.size(1) / gradG.size(2);
   const auto g_rows = width/dimx, g_cols = cols/dimy;

   printf("-> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(batches * g_rows, g_cols);
   dim3 block(1, 1);
   gmm_gradg_double2<<<grid, block>>>(gradG.data<double>(), X.data<double>(), W.data<double>(), gradY.data<double>(), batches, rows, cols, width, dimx, dimy);
}

__global__ void gmm_gradg_double3(double *gradG, double *X, double *W, double *gradY, int batches, int rows, int cols, int width, int dimx, int dimy) {
   const auto x_rows = rows, x_cols = width;
   const auto g_rows = width/dimx, g_cols = cols/dimy;
   const auto y_rows = rows, y_cols = cols;
   const auto w_cols = cols;

   const auto b = blockIdx.x / g_rows;
   const auto gr = blockIdx.x % g_rows;
   const auto gc = blockIdx.y;

   double tmp = 0;
   for (int i = 0; i < dimx; i++) {
      for (int j = 0; j < dimy; j++) {
         int k = gr * dimx + i;
         int c = gc * dimy + j;
         for (auto r = 0; r < rows; r++) {
            tmp += X0(b,r,k) * W0(k,c) * dY(b,r,c);
         }
      }
   }
   dG(b,gr,gc) = tmp;
}

void cuda_gmm_backward_g3(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != gradG.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

   const auto batches = gradY.size(0);
   const auto rows = gradY.size(1);
   const auto cols = gradY.size(2);
   const auto width = W.size(0);

   const auto dimx = W.size(0) / gradG.size(1);
   const auto dimy = W.size(1) / gradG.size(2);
   const auto g_rows = width/dimx, g_cols = cols/dimy;

   printf("cuda_gmm_backward_g3 -> batches=%ld rows=%ld cols=%ld width=%ld dimx=%ld dimy=%ld\n", batches, rows, cols, width, dimx, dimy);

   dim3 grid(batches * g_rows, g_cols);
   dim3 block(1, 1);
   gmm_gradg_double3<<<grid, block>>>(gradG.data<double>(), X.data<double>(), W.data<double>(), gradY.data<double>(), batches, rows, cols, width, dimx, dimy);
}
