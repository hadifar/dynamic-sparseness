#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
void cuda_gmm1(torch::Tensor Y, torch::Tensor A, torch::Tensor X, torch::Tensor G);
void cuda_gmm1_backward_g(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY);
void cuda_gmm2_backward_g(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY);
void cuda_gmm1_backward_w(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G);
void cuda_gmm_forward1(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);
void cuda_gmm_forward2(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);
void cuda_gmm_forward3(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);
void cuda_gmm_backward_w1(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G);
void cuda_gmm_backward_w2(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G);
void cuda_gmm_backward_w3(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G);
void cuda_gmm_backward_g1(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY);
void cuda_gmm_backward_g2(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY);
void cuda_gmm_backward_g3(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY);


// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void cuda_mm1(torch::Tensor y, torch::Tensor A, torch::Tensor w);
void mm1(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != W.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(1) != W.size(0)) AT_ERROR("invalid matrix dimensions");

   cuda_mm1(Y, X, W);
}

void cuda_mm4(torch::Tensor y, torch::Tensor A, torch::Tensor w);
void mm4(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != W.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(1) != W.size(0)) AT_ERROR("invalid matrix dimensions");

   cuda_mm4(Y, X, W);
}

void cuda_mm5(torch::Tensor y, torch::Tensor A, torch::Tensor w);
void mm5(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != W.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(1) != W.size(0)) AT_ERROR("invalid matrix dimensions");

   cuda_mm5(Y, X, W);
}

void cuda_mm5_TN(torch::Tensor y, torch::Tensor A, torch::Tensor w);
void mm5_TN(torch::Tensor Y, torch::Tensor X, torch::Tensor W) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   if (Y.size(0) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != W.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(0) != W.size(0)) AT_ERROR("invalid matrix dimensions");

   cuda_mm5_TN(Y, X, W);
}

//void gmm_backward_g(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
//   CHECK_INPUT(gradG);
//   CHECK_INPUT(X);
//   CHECK_INPUT(W);
//   CHECK_INPUT(gradY);
//
//   cuda_gmm2_backward_g(gradG, X, W, gradY);
//}

//void gmm_backward_w(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
//   CHECK_INPUT(gradW);
//   CHECK_INPUT(X);
//   CHECK_INPUT(gradY);
//   CHECK_INPUT(G);
//
//   cuda_gmm1_backward_w(gradW, X, gradY, G);
//}

void gmm_forward_gpu(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   CHECK_INPUT(G);

   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

//   cuda_gmm_forward1(Y, X, W, G);
//   cuda_gmm_forward2(Y, X, W, G);
   cuda_gmm_forward3(Y, X, W, G);
}

void gmm_backward_w_gpu(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
   CHECK_INPUT(gradY);
   CHECK_INPUT(X);
   CHECK_INPUT(gradW);
   CHECK_INPUT(G);

   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != gradW.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != gradW.size(1)) AT_ERROR("invalid matrix dimensions");

//   cuda_gmm_backward_w1(gradW, X, gradY, G);
//   cuda_gmm_backward_w2(gradW, X, gradY, G);
    cuda_gmm_backward_w3(gradW, X, gradY, G);
}

void gmm_backward_g_gpu(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   CHECK_INPUT(gradG);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   CHECK_INPUT(gradY);

   if (gradY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(0) != gradG.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(1) != X.size(1)) AT_ERROR("invalid matrix dimensions");
   if (X.size(2) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (gradY.size(2) != W.size(1)) AT_ERROR("invalid matrix dimensions");

//   cuda_gmm_backward_g1(gradG, X, W, gradY);
//   cuda_gmm_backward_g2(gradG, X, W, gradY);
   cuda_gmm_backward_g3(gradG, X, W, gradY);
}


void gmm_forward_cpu(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   CHECK_INPUT(G);

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

   printf("batches=%ld rows=%ld cols=%ld width=%ld\n", batches, rows, cols, width);

   auto x = X.cpu();
   auto w = W.cpu();
   auto g = G.cpu();

   for (auto b = 0; b < batches; b++) {
     for (auto r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
           double tmp = 0;
           for (int k = 0; k < width; k++) {
              tmp += x[b][r][k].data<double>()[0] * w[k][c].data<double>()[0] * g[b][k/dimx][c/dimy].data<double>()[0];
           }
           Y[b][r][c] = tmp;
        }
     }
   }
}

void gmm_backward_w_cpu(torch::Tensor gradW, torch::Tensor X, torch::Tensor gradY, torch::Tensor G) {
   CHECK_INPUT(gradY);
   CHECK_INPUT(X);
   CHECK_INPUT(gradW);
   CHECK_INPUT(G);

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

   auto x = X.cpu();
   auto dy = gradY.cpu();
   auto g = G.cpu();

   for (int c = 0; c < cols; c++) {
      for (int k = 0; k < width; k++) {
         double tmp = 0;
         for (auto b = 0; b < batches; b++) {
           for (auto r = 0; r < rows; r++) {
              tmp += x[b][r][k].data<double>()[0] * g[b][k/dimx][c/dimy].data<double>()[0] * dy[b][r][c].data<double>()[0];
           }
         }
         gradW[k][c] += tmp;
      }
   }
}

void gmm_backward_g_cpu(torch::Tensor gradG, torch::Tensor X, torch::Tensor W, torch::Tensor gradY) {
   CHECK_INPUT(gradY);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   CHECK_INPUT(gradG);

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

   auto x = X.cpu();
   auto w = W.cpu();
   auto dy = gradY.cpu();

   for (auto b = 0; b < batches; b++) {
     for (auto r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
           for (int k = 0; k < width; k++) {
              gradG[b][k/dimx][c/dimy] += x[b][r][k].data<double>()[0] * w[k][c].data<double>()[0] * dy[b][r][c].data<double>()[0];
           }
        }
     }
   }
}


/*
 * GMV implementation
 */

void cuda_gmv_forward1(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);
void cuda_gmv_forward2(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);
void cuda_gmv_forward3(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);
void fast_gmv_forward(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G);

void gmv_forward_gpu(torch::Tensor Y, torch::Tensor X, torch::Tensor W, torch::Tensor G, int version) {
   CHECK_INPUT(Y);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   CHECK_INPUT(G);

   if (Y.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (X.size(1) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (Y.size(1) != W.size(1)) AT_ERROR("invalid matrix dimensions");

    if (version == 1) {
        cuda_gmv_forward1(Y, X, W, G);
    } else if (version == 2) {
        cuda_gmv_forward2(Y, X, W, G);
    } else if (version == 3) {
        cuda_gmv_forward3(Y, X, W, G);
    } else if (version == 4 || version == 5 || version == 6) {
        fast_gmv_forward(Y, X, W, G);
    }
}


void cuda_gmv_backward_g_1(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);
void cuda_gmv_backward_g_2(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);
void cuda_gmv_backward_g_3(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);
void fast_gmv_gradg(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);
void fast_gmv_gradg_2(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);
void fast_gmv_gradg_3(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);
void fast_gmv_gradg_4(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG);

void gmv_backward_g_gpu(torch::Tensor dY, torch::Tensor X, torch::Tensor W, torch::Tensor dG, int version) {
   CHECK_INPUT(dY);
   CHECK_INPUT(X);
   CHECK_INPUT(W);
   CHECK_INPUT(dG);

   if (dY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (dY.size(0) != dG.size(0)) AT_ERROR("invalid matrix dimensions");
   if (X.size(1) != W.size(0)) AT_ERROR("invalid matrix dimensions");
   if (dY.size(1) != W.size(1)) AT_ERROR("invalid matrix dimensions");

    if (version == 1) {
        cuda_gmv_backward_g_1(dY, X, W, dG);
    } else if (version == 2) {
        cuda_gmv_backward_g_2(dY, X, W, dG);
    } else if (version == 3) {
        cuda_gmv_backward_g_3(dY, X, W, dG);
    } else if (version == 4) {
        fast_gmv_gradg(dY, X, W, dG);
    } else if (version == 5) {
        fast_gmv_gradg_2(dY, X, W, dG);
    } else if (version == 6) {
        fast_gmv_gradg_3(dY, X, W, dG);
    } else if (version == 7) {
        fast_gmv_gradg_4(dY, X, W, dG);
    }
}


void cuda_gmv_backward_w_1(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G);
void cuda_gmv_backward_w_2(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G);
void cuda_gmv_backward_w_3(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G);
void fast_gmv_gradw(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G);

void gmv_backward_w_gpu(torch::Tensor dY, torch::Tensor X, torch::Tensor dW, torch::Tensor G, int version) {
   CHECK_INPUT(dY);
   CHECK_INPUT(X);
   CHECK_INPUT(dW);
   CHECK_INPUT(G);

   if (dY.size(0) != X.size(0)) AT_ERROR("invalid matrix dimensions");
   if (dY.size(0) != G.size(0)) AT_ERROR("invalid matrix dimensions");
   if (X.size(1) != dW.size(0)) AT_ERROR("invalid matrix dimensions");
   if (dY.size(1) != dW.size(1)) AT_ERROR("invalid matrix dimensions");

    if (version == 1) {
        cuda_gmv_backward_w_1(dY, X, dW, G);
    } else if (version == 2) {
        cuda_gmv_backward_w_2(dY, X, dW, G);
    } else if (version == 3) {
        cuda_gmv_backward_w_3(dY, X, dW, G);
    } else if (version == 4 || version == 5 || version == 6) {
        fast_gmv_gradw(dY, X, dW, G);
    }
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mm1", &mm1, "custom matrix multiplication 1");
    m.def("mm4", &mm4, "custom matrix multiplication 4");
    m.def("mm5", &mm5, "custom matrix multiplication 5");
    m.def("mm5_TN", &mm5_TN, "custom matrix multiplication 5");

//  m.def("mymm2", &mymm2, "custom matrix multiplication 2");
//  m.def("mymm3", &mymm3, "custom matrix multiplication 3");
//  m.def("mymm4", &mymm4, "custom matrix multiplication 4");
//  m.def("gated_mm_cpu", &gated_mm_cpu, "gated matrix multiplication cpu");
//  m.def("gmm_backward_g", &gmm_backward_g, "gated matrix multiplication cpu backward");
//  m.def("gmm_backward_w", &gmm_backward_w, "gated matrix multiplication cpu backward");

  m.def("gmm_forward_cpu", &gmm_forward_cpu, "gated matrix multiplication cpu");
  m.def("gmm_backward_g_cpu", &gmm_backward_g_cpu, "gated matrix multiplication cpu");
  m.def("gmm_backward_w_cpu", &gmm_backward_w_cpu, "gated matrix multiplication cpu");

  m.def("gmm_forward_gpu", &gmm_forward_gpu, "gated matrix multiplication gpu");
  m.def("gmm_backward_g_gpu", &gmm_backward_g_gpu, "gated matrix multiplication gpu");
  m.def("gmm_backward_w_gpu", &gmm_backward_w_gpu, "gated matrix multiplication gpu");

  m.def("gmv_forward_gpu", &gmv_forward_gpu, "gated matrix vector multiplication gpu");
  m.def("gmv_backward_g_gpu", &gmv_backward_g_gpu, "gated matrix vector multiplication gpu");
  m.def("gmv_backward_w_gpu", &gmv_backward_w_gpu, "gated matrix vector multiplication gpu");
}
