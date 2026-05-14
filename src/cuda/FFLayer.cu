#include "FFLayer.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include "Stream.hpp"

namespace
{

    void check_cuda(cudaError_t err, const char *msg)
    {
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }

    void check_cublas(cublasStatus_t status, const char *msg)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::string(msg));
        }
    }

    cublasHandle_t get_cublas_handle()
    {
        static cublasHandle_t handle = nullptr;
        static bool initialized = false;

        if (!initialized)
        {
            check_cublas(cublasCreate(&handle), "cublasCreate failed");
            cudaStream_t stream = get_cuda_stream();
            cublasSetStream(handle, stream);
            initialized = true;
        }
        return handle;
    }

    __global__ void add_bias_kernel(float *z, const float *b, int rows, int cols)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int total = rows * cols;
        if (idx < total)
        {
            int col = idx % cols;
            z[idx] += b[col];
        }
    }

    __global__ void relu_forward_kernel(const float *z, float *a, int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            a[idx] = z[idx] > 0.0f ? z[idx] : 0.0f;
        }
    }

    __global__ void relu_backward_kernel(const float *z,
                                         const float *grad_out,
                                         float *grad_z,
                                         int n)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n)
        {
            grad_z[idx] = z[idx] > 0.0f ? grad_out[idx] : 0.0f;
        }
    }

    __global__ void sum_rows_kernel(const float *x, float *out, int rows, int cols)
    {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < cols)
        {
            float sum = 0.0f;
            for (int r = 0; r < rows; ++r)
            {
                sum += x[r * cols + col];
            }
            out[col] = sum;
        }
    }

} // namespace

FFLayer::FFLayer(std::string name,
                 Activation activation,
                 int output_size,
                 int input_size)
    : name_(std::move(name)),
      activation_(activation),
      output_size_(output_size),
      input_size_(input_size),
      W_(input_size, output_size),
      b_(1, output_size),
      dW_(input_size, output_size),
      db_(1, output_size),
      dW_accum_(input_size, output_size),
      db_accum_(1, output_size) {}

void FFLayer::init(unsigned int seed)
{
    std::mt19937 rng(seed);

    std::vector<float> host_w(W_.size());
    std::vector<float> host_b(b_.size(), 0.01f);

    float sigma = (activation_ == Activation::ReLU)
                      ? std::sqrt(2.0f / static_cast<float>(input_size_))
                      : std::sqrt(1.0f / static_cast<float>(input_size_));

    std::normal_distribution<float> dist(0.0f, sigma);

    for (std::size_t i = 0; i < host_w.size(); ++i)
    {
        host_w[i] = dist(rng);
    }

    W_.copy_from_host(host_w);
    b_.copy_from_host(host_b);

    dW_.zero();
    db_.zero();
}

void FFLayer::forward(const Tensor &input, Tensor &output)
{
    int batch = input.rows();

    input_cache_.resize_like(input);
    input_cache_.copy_from_device(input, get_cuda_stream());

    preact_cache_.resize(batch, output_size_);
    output_cache_.resize(batch, output_size_);

    // preact_cache_ = input * W_
    // Shapes:
    // input:  [batch, input_size_]
    // W_:     [input_size_, output_size_]
    // output: [batch, output_size_]

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // cuBLAS is column-major by default. We can treat row-major matrices by
    // swapping operand order appropriately:
    // C(row-major) = A(row-major) * B(row-major)
    // -> cublasSgemm computes C^T = B^T * A^T
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            output_size_, // m
            batch,        // n
            input_size_,  // k
            &alpha,
            W_.data(), output_size_,
            input.data(), input_size_,
            &beta,
            preact_cache_.data(), output_size_),
        "cublasSgemm forward failed");

    int total = batch * output_size_;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    add_bias_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
        preact_cache_.data(),
        b_.data(),
        batch,
        output_size_);
    check_cuda(cudaGetLastError(), "add_bias_kernel launch failed");

    if (activation_ == Activation::ReLU)
    {
        relu_forward_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
            preact_cache_.data(),
            output_cache_.data(),
            total);
        check_cuda(cudaGetLastError(), "relu_forward_kernel launch failed");
    }
    else
    {
        output_cache_.copy_from_device(preact_cache_, get_cuda_stream());
    }

    output.resize_like(output_cache_);
    output.copy_from_device(output_cache_, get_cuda_stream());
}

void FFLayer::backward_parameter_grads(const Tensor &grad_output)
{
    int batch = grad_output.rows();

    act_grad_cache_.resize_like(grad_output);

    int total = batch * output_size_;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    if (activation_ == Activation::ReLU)
    {
        relu_backward_kernel<<<blocks, threads, 0, get_cuda_stream()>>>(
            preact_cache_.data(),
            grad_output.data(),
            act_grad_cache_.data(),
            total);
        check_cuda(cudaGetLastError(), "relu_backward_kernel launch failed");
    }
    else
    {
        act_grad_cache_.copy_from_device(grad_output, get_cuda_stream());
    }

    // db_ = sum_rows(act_grad_cache_)
    {
        int bias_threads = 256;
        int bias_blocks = (output_size_ + bias_threads - 1) / bias_threads;

        sum_rows_kernel<<<bias_blocks, bias_threads, 0, get_cuda_stream()>>>(
            act_grad_cache_.data(),
            db_.data(),
            batch,
            output_size_);
        check_cuda(cudaGetLastError(), "sum_rows_kernel launch failed");
    }

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // dW_ = input_cache_^T * act_grad_cache_
    // Shapes:
    // input_cache_:     [batch, input_size_]
    // act_grad_cache_:  [batch, output_size_]
    // dW_:              [input_size_, output_size_]
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            output_size_, // m
            input_size_,  // n
            batch,        // k
            &alpha,
            act_grad_cache_.data(), output_size_,
            input_cache_.data(), input_size_,
            &beta,
            dW_.data(), output_size_),
        "cublasSgemm dW failed");
}

void FFLayer::zero_grad_accum(cudaStream_t stream)
{
    check_cuda(cudaMemsetAsync(dW_accum_.data(), 0, dW_accum_.bytes(), stream),
               "cudaMemsetAsync dW_accum_ failed");
    check_cuda(cudaMemsetAsync(db_accum_.data(), 0, db_accum_.bytes(), stream),
               "cudaMemsetAsync db_accum_ failed");
}

void FFLayer::accumulate_grads(cudaStream_t stream)
{
    dW_accum_.accumulate_slice(dW_, 0, 0, dW_.size(), stream);
    db_accum_.accumulate_slice(db_, 0, 0, db_.size(), stream);
}

void FFLayer::backward_input_grad(Tensor &grad_input)
{
    int batch = act_grad_cache_.rows();
    grad_input.resize(batch, input_size_);

    cublasHandle_t handle = get_cublas_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // grad_input = act_grad_cache_ * W_^T
    // Shapes:
    // act_grad_cache_: [batch, output_size_]
    // W_^T:            [output_size_, input_size_]
    // grad_input:      [batch, input_size_]
    check_cublas(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            input_size_,  // m
            batch,        // n
            output_size_, // k
            &alpha,
            W_.data(), output_size_,
            act_grad_cache_.data(), output_size_,
            &beta,
            grad_input.data(), input_size_),
        "cublasSgemm grad_input failed");
}
