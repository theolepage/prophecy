#include "cuda_memory.cuh"

void cudaXMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
{
    cuda_safe_call(cudaMemcpy(dst, src, count, kind));
}

void cudaXMemcpyAsync(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream)
{
    cuda_safe_call(cudaMemcpyAsync(dst, src, count, kind, stream));
}

void cudaXMemset(void* devPtr, int  value, size_t count)
{
    cuda_safe_call(cudaMemset(devPtr, value, count));
}

void cudaXFree(void* devPtr)
{
    cuda_safe_call(cudaFree(devPtr));
}