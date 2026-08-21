import os
from torch.utils.cpp_extension import load
os.environ['MAX_JOBS']='1'
load(name='rwkv7_dense_gemv_mapping_probe', sources=['/tmp/rwkv7_dense_gemv_mapping_probe/gemv_variants.cu'], with_cuda=True, is_python_module=False, build_directory='/tmp/rwkv7_dense_gemv_mapping_probe/build', extra_cuda_cflags=['-O3'])
