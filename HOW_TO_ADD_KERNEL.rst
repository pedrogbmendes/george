Adding a new kernel requires a few steps:

1. Implement the kernel in the C++ header file 'include/kernels.h'
2. Add the class to the Cython definition file 'george/kernels.pxd'
3. Define a corresponding Python Class in 'george/kernels.py' with a unique kernel_type
4. add this kernel_type to the function 'parse_kernel(kernel_spec)' in 'george/kernels.pxd'
