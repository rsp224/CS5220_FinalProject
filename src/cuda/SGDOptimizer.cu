#include "SGDOptimizer.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

__global__ void sgd_update_kernel(float* param,
                                  float* grad,
                                  std::size_t n,
                                  float learning_rate) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        param[idx] -= learning_rate * grad[idx];
        grad[idx] = 0.0f;
    }
}

void update_tensor(Tensor& param, Tensor& grad, float learning_rate) {
    if (param.size() != grad.size()) {
        throw std::runtime_error("SGDOptimizer: parameter and gradient size mismatch.");
    }

    const int threads = 256;
    const int blocks = static_cast<int>((param.size() + threads - 1) / threads);

    sgd_update_kernel<<<blocks, threads>>>(
        param.data(),
        grad.data(),
        param.size(),
        learning_rate
    );
    check_cuda(cudaGetLastError(), "sgd_update_kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "sgd_update_kernel sync failed");
}

}  // namespace

SGDOptimizer::SGDOptimizer(float learning_rate)
    : learning_rate_(learning_rate) {}

void SGDOptimizer::step(FFLayer& layer) {
    update_tensor(layer.weights(), layer.weight_grads(), learning_rate_);
    update_tensor(layer.biases(), layer.bias_grads(), learning_rate_);
}

float SGDOptimizer::learning_rate() const noexcept {
    return learning_rate_;
}