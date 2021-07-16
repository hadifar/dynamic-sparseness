from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_cuda',
    packages=['tcop'],
    ext_modules=[
        CUDAExtension('lltm_cuda', [
            'src/lltm_cuda.cpp',
            'src/lltm_cuda_kernel.cu',
            'src/gmm.cu',
            'src/gmv.cu',
            'src/matmul.cu',
            'src/fast_gmv.cu',
        ],
        library_dirs=['objs'],
        extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O3', '-gencode', 'arch=compute_61,code=sm_61', '-Xptxas', '-O3,-v']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
