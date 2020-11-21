#include "kernel.cuh"

#include <cuda_runtime.h>
#include <stdio.h>

__global__
void func()
{
    printf("Hello world!\n");
}

void kernel()
{
    func<<<1, 1>>>();
    cudaDeviceSynchronize();
}