#include "Stream.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

void check_cuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " +
                                 cudaGetErrorString(err));
    }
}

cudaStream_t get_cuda_stream()
{
    static cudaStream_t stream;
    static bool initialized = false;

    if (!initialized)
    {
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate failed");
        initialized = true;
    }
    return stream;
}

cudaStream_t get_comm_stream()
{
    static cudaStream_t stream;
    static bool initialized = false;

    if (!initialized)
    {
        check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate (comm) failed");
        initialized = true;
    }
    return stream;
}