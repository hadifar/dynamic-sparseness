#include <torch/types.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TILE_SIZE 8

#define X0(a,b)     X[(a)*x_cols+(b)]
#define Y0(a,b)     Y[(a)*y_cols+(b)]
#define W0(a,b)     W[(a)*w_cols+(b)]
#define G0(a,b,c)   G[(a)*g_rows*g_cols+(b)*g_cols+(c)]

#define DY(a,b)     dY[(a)*y_cols+(b)]
#define DG(a,b,c)   dG[(a)*g_rows*g_cols+(b)*g_cols+(c)]
#define DW(a,b)     dW[(a)*w_cols+(b)]


/*
 * Forward functions
 */

__global__ void gmv_double1(double *Y, double *X, double *W, double *G, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    for (auto b = 0; b < batches; b++) {
        for (auto o = 0; o < dim_output; o++) {
            double tmp = 0.0;
            for (auto i = 0; i < dim_input; i++) {
                tmp += X0(b, i) * W0(i, o) * G0(b,i/dimx,o/dimy);
            }
            Y0(b,o) = tmp;
        }
    }
}

void cuda_gmv_forward1(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   const auto batches = Y.size(0);
   const auto dim_output = Y.size(1);
   const auto dim_input = X.size(1);

   const auto dimx = W.size(0) / G.size(1);
   const auto dimy = W.size(1) / G.size(2);

   printf("cuda_gmv_forward1 -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);

   dim3 grid(1, 1);
   dim3 block(1, 1);
   gmv_double1<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), batches, dim_output, dim_input, dimx, dimy);
}

__global__ void gmv_double2(double *Y, double *X, double *W, double *G, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto b = blockIdx.x;
    const auto o = blockIdx.y;
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    double tmp = 0.0;
    for (auto i = 0; i < dim_input; i++) {
        tmp += X0(b, i) * W0(i, o) * G0(b,i/dimx,o/dimy);
    }
    Y0(b,o) = tmp;
}

void cuda_gmv_forward2(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   const auto batches = Y.size(0);
   const auto dim_output = Y.size(1);
   const auto dim_input = X.size(1);

   const auto dimx = W.size(0) / G.size(1);
   const auto dimy = W.size(1) / G.size(2);

   printf("cuda_gmv_forward2 -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);

   dim3 grid(batches, dim_output);
   dim3 block(1, 1);
   gmv_double2<<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), batches, dim_output, dim_input, dimx, dimy);
}


template<typename T> __global__ void gmv_3(T *Y, T *X, T *W, T *G, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto b0 = blockIdx.x * blockDim.x;
    const auto b = threadIdx.x;
    const auto o0 = blockIdx.y * blockDim.y;
    const auto o = threadIdx.y;
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    T tmp = 0.0;
    int i0 = 0;
    for (auto m = 0; m < dim_input / TILE_SIZE; m++) {
        __shared__ T Xs[TILE_SIZE][TILE_SIZE];
        __shared__ T Ws[TILE_SIZE][TILE_SIZE];
        Xs[threadIdx.y][threadIdx.x] = X0(b0+threadIdx.y, i0+threadIdx.x);
        Ws[threadIdx.y][threadIdx.x] = W0(i0+threadIdx.y, o0+threadIdx.x);

        __syncthreads();
        T tmp1 = 0.0;
        for (auto i = 0; i < TILE_SIZE; i++) {
//            tmp += X0(b0+b, i0 + i) * W0(i0 + i, o0+o) * G0(b0+b,(i0+i)/dimx,(o0+o)/dimy);
            tmp1 += Xs[b][i] * Ws[i][o];
        }
        tmp += tmp1 * G0(b0+b,i0/dimx,o0/dimy);
        i0 += TILE_SIZE;
        __syncthreads();
    }
    Y0(b0+b,o0+o) = tmp;
}

void cuda_gmv_forward3(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
    const auto batches = Y.size(0);
    const auto dim_output = Y.size(1);
    const auto dim_input = X.size(1);

    const auto dimx = W.size(0) / G.size(1);
    const auto dimy = W.size(1) / G.size(2);

    if (batches % TILE_SIZE != 0) AT_ERROR("respect tile size for batches");
    if (dim_output % TILE_SIZE != 0) AT_ERROR("respect tile size for dim_output");

//   printf("cuda_gmv_forward3 -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);

    dim3 grid(batches / TILE_SIZE, dim_output / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    if (Y.type().scalarType() == torch::ScalarType::Double) {
        gmv_3<double><<<grid, block>>>(Y.data<double>(), X.data<double>(), W.data<double>(), G.data<double>(), batches, dim_output, dim_input, dimx, dimy);
    } else {
        printf("float\n");
        gmv_3<float><<<grid, block>>>(Y.data<float>(), X.data<float>(), W.data<float>(), G.data<float>(), batches, dim_output, dim_input, dimx, dimy);
    }
}


/*
 * Backward_g functions
 */

__global__ void gmv_backward_g_1_double(double *dY, double *X, double *W, double *dG, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    for (auto b = 0; b < batches; b++) {
        for (auto o = 0; o < dim_output; o++) {
            for (auto i = 0; i < dim_input; i++) {
                DG(b,i/dimx,o/dimy) += X0(b, i) * W0(i, o) * DY(b,o);
            }
        }
    }
}

void cuda_gmv_backward_g_1(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
   const auto batches = dY.size(0);
   const auto dim_output = dY.size(1);
   const auto dim_input = X.size(1);

   const auto dimx = W.size(0) / dG.size(1);
   const auto dimy = W.size(1) / dG.size(2);

   printf("cuda_gmv_backward_g_1 -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);

   dim3 grid(1, 1);
   dim3 block(1, 1);
   gmv_backward_g_1_double<<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), batches, dim_output, dim_input, dimx, dimy);
}


__global__ void gmv_backward_g_2_double(double *dY, double *X, double *W, double *dG, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    const auto b = blockIdx.x / g_rows;
    const auto gr = blockIdx.x % g_rows;
    const auto gc = blockIdx.y;

    double tmp = 0;
    for (int u = 0; u < dimx; u++) {
        for (int v = 0; v < dimy; v++) {
            int i = gr * dimx + u;
            int o = gc * dimy + v;
            tmp += X0(b, i) * W0(i, o) * DY(b,o);
        }
    }
    DG(b,gr,gc) = tmp;
}

void cuda_gmv_backward_g_2(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
   const auto batches = dY.size(0);
   const auto dim_output = dY.size(1);
   const auto dim_input = X.size(1);

   const auto dimx = W.size(0) / dG.size(1);
   const auto dimy = W.size(1) / dG.size(2);
   const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

   printf("cuda_gmv_backward_g_2 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

   dim3 grid(batches*g_rows, g_cols);
   dim3 block(1, 1);
   gmv_backward_g_2_double<<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), batches, dim_output, dim_input, dimx, dimy);
}


template<typename T> __global__ void gmv_backward_g_3(T *dY, T *X, T *W, T *dG, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    const auto b = blockIdx.x / g_rows;
    const auto gr = blockIdx.x % g_rows;
    const auto gc = blockIdx.y;

    T tmp = 0;
    for (int u = 0; u < dimx; u++) {
        int i = gr * dimx + u;
        int o = gc * dimy;

        T tmp1 = 0;
        for (int v = 0; v < dimy; v++) {
            tmp1 += W0(i, o) * DY(b,o);
            o++;
        }
        tmp += X0(b, i) * tmp1;
    }
    DG(b,gr,gc) = tmp;
}

void cuda_gmv_backward_g_3(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG) {
   const auto batches = dY.size(0);
   const auto dim_output = dY.size(1);
   const auto dim_input = X.size(1);

   const auto dimx = W.size(0) / dG.size(1);
   const auto dimy = W.size(1) / dG.size(2);
   const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

   printf("cuda_gmv_backward_g_3 -> batches=%ld dim_output=%ld dim_input=%ld blocks=%ld,%ld,%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, batches, g_rows, g_cols, dimx, dimy);

   dim3 grid(batches*g_rows, g_cols);
   dim3 block(1, 1);
   if (dY.type().scalarType() == torch::ScalarType::Double) {
       gmv_backward_g_3<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), W.data<double>(), dG.data<double>(), batches, dim_output, dim_input, dimx, dimy);
   } else {
      gmv_backward_g_3<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), W.data<float>(), dG.data<float>(), batches, dim_output, dim_input, dimx, dimy);
   }
}


/*
 * Backward_w functions
 */


__global__ void gmv_backward_w_1_double(double *dY, double *X, double *dW, double *G, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    for (auto b = 0; b < batches; b++) {
        for (auto o = 0; o < dim_output; o++) {
            for (auto i = 0; i < dim_input; i++) {
                DW(i, o) += X0(b, i) * G0(b,i/dimx,o/dimy) * DY(b,o);
            }
        }
    }
}

void cuda_gmv_backward_w_1(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G) {
    const auto batches = dY.size(0);
    const auto dim_output = dY.size(1);
    const auto dim_input = X.size(1);

    const auto dimx = dW.size(0) / G.size(1);
    const auto dimy = dW.size(1) / G.size(2);

    printf("cuda_gmv_backward_g_1 -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);

    dim3 grid(1, 1);
    dim3 block(1, 1);
    gmv_backward_w_1_double<<<grid, block>>>(dY.data<double>(), X.data<double>(), dW.data<double>(), G.data<double>(), batches, dim_output, dim_input, dimx, dimy);
}

__global__ void gmv_backward_w_2_double(double *dY, double *X, double *dW, double *G, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto o = blockIdx.x;
    const auto i = blockIdx.y;
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    double tmp = 0;
    for (auto b = 0; b < batches; b++) {
        tmp += X0(b, i) * G0(b,i/dimx,o/dimy) * DY(b,o);
    }
    DW(i, o) = tmp;
}

void cuda_gmv_backward_w_2(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G) {
    const auto batches = dY.size(0);
    const auto dim_output = dY.size(1);
    const auto dim_input = X.size(1);

    const auto dimx = dW.size(0) / G.size(1);
    const auto dimy = dW.size(1) / G.size(2);

    printf("gmv_backward_w_2_double -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);

    dim3 grid(dim_output, dim_input);
    dim3 block(1, 1);
    gmv_backward_w_2_double<<<grid, block>>>(dY.data<double>(), X.data<double>(), dW.data<double>(), G.data<double>(), batches, dim_output, dim_input, dimx, dimy);
}


template<typename T> __global__ void gmv_backward_w_3(T *dY, T *X, T *dW, T *G, int batches, int dim_output, int dim_input, int dimx, int dimy) {
    const auto o0 = blockIdx.x * blockDim.x, o = threadIdx.x;
    const auto i0 = blockIdx.y * blockDim.y, i = threadIdx.y;
    const auto y_rows = batches, y_cols = dim_output;
    const auto x_rows = batches, x_cols = dim_input;
    const auto w_rows = dim_input, w_cols = dim_output;
    const auto g_rows = dim_input/dimx, g_cols = dim_output/dimy;

    T tmp = 0;
    int b0 = 0;
    for (auto m = 0; m < batches / TILE_SIZE; m++) {
        __shared__ T Xs[TILE_SIZE][TILE_SIZE];
        __shared__ T Ys[TILE_SIZE][TILE_SIZE];
        Xs[threadIdx.x][threadIdx.y] = X0(b0+threadIdx.x, i0+threadIdx.y);
        Ys[threadIdx.x][threadIdx.y] = DY(b0+threadIdx.x, o0+threadIdx.y);

        // todo: bring G0 into shared mem?
        __syncthreads();
        for (auto b = 0; b < TILE_SIZE; b++) {
//            tmp += X0(b0+b, i0+i) * G0(b0+b,(i0+i)/dimx,(o0+o)/dimy) * DY(b0+b,o0+o);
            tmp += Xs[b][i] * Ys[b][o] * G0(b0+b,(i0+i)/dimx,(o0+o)/dimy);
        }
        b0 += TILE_SIZE;
        __syncthreads();
    }
    DW(i0+i, o0+o) = tmp;
}

void cuda_gmv_backward_w_3(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G) {
    const auto batches = dY.size(0);
    const auto dim_output = dY.size(1);
    const auto dim_input = X.size(1);

    const auto dimx = dW.size(0) / G.size(1);
    const auto dimy = dW.size(1) / G.size(2);

//    printf("gmv_backward_w_3_double -> batches=%ld dim_output=%ld dim_input=%ld dimx=%ld dimy=%ld\n", batches, dim_output, dim_input, dimx, dimy);



    if (dim_output % TILE_SIZE != 0) AT_ERROR("respect tile size for dim_output");
    if (dim_input % TILE_SIZE != 0) AT_ERROR("respect tile size for dim_input");

    dim3 grid(dim_output / TILE_SIZE, dim_input / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    if (dY.type().scalarType() == torch::ScalarType::Double) {
        gmv_backward_w_3<double><<<grid, block>>>(dY.data<double>(), X.data<double>(), dW.data<double>(), G.data<double>(), batches, dim_output, dim_input, dimx, dimy);
    } else {
        gmv_backward_w_3<float><<<grid, block>>>(dY.data<float>(), X.data<float>(), dW.data<float>(), G.data<float>(), batches, dim_output, dim_input, dimx, dimy);
    }
}
