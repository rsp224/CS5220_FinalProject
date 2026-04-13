#include "Tensor.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

__global__ void zero_kernel(float* data, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

}  // namespace

Tensor::Tensor()
    : data_(nullptr), rows_(0), cols_(0), size_(0) {}

Tensor::Tensor(int rows, int cols)
    : data_(nullptr), rows_(0), cols_(0), size_(0) {
    resize(rows, cols);
}

Tensor::~Tensor() {
    free();
}

Tensor::Tensor(Tensor&& other) noexcept
    : data_(other.data_),
      rows_(other.rows_),
      cols_(other.cols_),
      size_(other.size_) {
    other.data_ = nullptr;
    other.rows_ = 0;
    other.cols_ = 0;
    other.size_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        free();

        data_ = other.data_;
        rows_ = other.rows_;
        cols_ = other.cols_;
        size_ = other.size_;

        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
        other.size_ = 0;
    }
    return *this;
}

void Tensor::resize(int rows, int cols) {
    if (rows <= 0 || cols <= 0) {
        throw std::runtime_error("Tensor dimensions must be positive.");
    }

    const std::size_t new_size =
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);

    if (rows == rows_ && cols == cols_ && data_ != nullptr) {
        return;
    }

    free();

    rows_ = rows;
    cols_ = cols;
    size_ = new_size;

    check_cuda(cudaMalloc(&data_, bytes()), "cudaMalloc failed");
}

void Tensor::resize_like(const Tensor& other) {
    resize(other.rows_, other.cols_);
}

void Tensor::zero() {
    if (data_ == nullptr) {
        throw std::runtime_error("Cannot zero an unallocated tensor.");
    }

    const int threads = 256;
    const int blocks = static_cast<int>((size_ + threads - 1) / threads);

    zero_kernel<<<blocks, threads>>>(data_, size_);
    check_cuda(cudaGetLastError(), "zero_kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "zero_kernel sync failed");
}

void Tensor::copy_from_host(const std::vector<float>& host_data) {
    if (data_ == nullptr) {
        throw std::runtime_error("Cannot copy into an unallocated tensor.");
    }
    if (host_data.size() != size_) {
        throw std::runtime_error("Host data size does not match tensor size.");
    }

    check_cuda(
        cudaMemcpy(data_, host_data.data(), bytes(), cudaMemcpyHostToDevice),
        "cudaMemcpy host to device failed"
    );
}

std::vector<float> Tensor::copy_to_host() const {
    if (data_ == nullptr) {
        throw std::runtime_error("Cannot copy from an unallocated tensor.");
    }

    std::vector<float> host_data(size_);

    check_cuda(
        cudaMemcpy(host_data.data(), data_, bytes(), cudaMemcpyDeviceToHost),
        "cudaMemcpy device to host failed"
    );

    return host_data;
}

void Tensor::copy_from_device(const Tensor& other) {
    if (other.data_ == nullptr) {
        throw std::runtime_error("Source tensor is unallocated.");
    }

    resize(other.rows_, other.cols_);

    check_cuda(
        cudaMemcpy(data_, other.data_, bytes(), cudaMemcpyDeviceToDevice),
        "cudaMemcpy device to device failed"
    );
}

float* Tensor::data() noexcept {
    return data_;
}

const float* Tensor::data() const noexcept {
    return data_;
}

int Tensor::rows() const noexcept {
    return rows_;
}

int Tensor::cols() const noexcept {
    return cols_;
}

std::size_t Tensor::size() const noexcept {
    return size_;
}

std::size_t Tensor::bytes() const noexcept {
    return size_ * sizeof(float);
}

void Tensor::free() {
    if (data_ != nullptr) {
        cudaFree(data_);
        data_ = nullptr;
    }
    rows_ = 0;
    cols_ = 0;
    size_ = 0;
}