#include "CCELoss.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// One block per row.
// This is simple and fine for MNIST-scale output size (10 classes).
__global__ void softmax_forward_kernel(const float* logits,
                                       float* probs,
                                       int rows,
                                       int cols) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    const float* row_logits = logits + row * cols;
    float* row_probs = probs + row * cols;

    float max_val = row_logits[0];
    for (int j = 1; j < cols; ++j) {
        if (row_logits[j] > max_val) {
            max_val = row_logits[j];
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < cols; ++j) {
        float e = expf(row_logits[j] - max_val);
        row_probs[j] = e;
        sum_exp += e;
    }

    for (int j = 0; j < cols; ++j) {
        row_probs[j] /= sum_exp;
    }
}

__global__ void cce_backward_kernel(const float* probs,
                                    const float* target,
                                    float* grad_logits,
                                    int total_size,
                                    float inv_batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        grad_logits[idx] = (probs[idx] - target[idx]) * inv_batch_size;
    }
}

}  // namespace

CCELoss::CCELoss(int num_classes, int batch_size)
    : num_classes_(num_classes),
      batch_size_(batch_size),
      inv_batch_size_(1.0f / static_cast<float>(batch_size)),
      probs_(batch_size, num_classes),
      target_(batch_size, num_classes),
      loss_(0.0f),
      cumulative_loss_(0.0f),
      correct_(0),
      incorrect_(0) {}

float CCELoss::forward(const Tensor& logits, const Tensor& target) {
    if (logits.rows() != batch_size_ || logits.cols() != num_classes_) {
        throw std::runtime_error("CCELoss::forward logits shape mismatch.");
    }
    if (target.rows() != batch_size_ || target.cols() != num_classes_) {
        throw std::runtime_error("CCELoss::forward target shape mismatch.");
    }

    target_.copy_from_device(target);

    softmax_forward_kernel<<<batch_size_, 1>>>(
        logits.data(),
        probs_.data(),
        batch_size_,
        num_classes_
    );
    check_cuda(cudaGetLastError(), "softmax_forward_kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "softmax_forward_kernel sync failed");

    std::vector<float> host_probs = probs_.copy_to_host();
    std::vector<float> host_target = target_.copy_to_host();

    loss_ = 0.0f;

    for (int i = 0; i < batch_size_; ++i) {
        int pred_class = 0;
        float pred_max = host_probs[i * num_classes_];

        int true_class = 0;

        for (int j = 0; j < num_classes_; ++j) {
            float p = host_probs[i * num_classes_ + j];
            float t = host_target[i * num_classes_ + j];

            if (p > pred_max) {
                pred_max = p;
                pred_class = j;
            }

            if (t > 0.0f) {
                true_class = j;
                loss_ -= logf(fmaxf(p, std::numeric_limits<float>::epsilon()));
            }
        }

        if (pred_class == true_class) {
            ++correct_;
        } else {
            ++incorrect_;
        }
    }

    loss_ *= inv_batch_size_;
    cumulative_loss_ += loss_;

    return loss_;
}

void CCELoss::backward(Tensor& grad_logits) {
    grad_logits.resize(batch_size_, num_classes_);

    int total = batch_size_ * num_classes_;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    cce_backward_kernel<<<blocks, threads>>>(
        probs_.data(),
        target_.data(),
        grad_logits.data(),
        total,
        inv_batch_size_
    );
    check_cuda(cudaGetLastError(), "cce_backward_kernel launch failed");
    check_cuda(cudaDeviceSynchronize(), "cce_backward_kernel sync failed");
}

float CCELoss::avg_loss() const noexcept {
    std::size_t total = correct_ + incorrect_;
    if (total == 0) {
        return 0.0f;
    }
    return cumulative_loss_ / static_cast<float>(total);
}

float CCELoss::accuracy() const noexcept {
    std::size_t total = correct_ + incorrect_;
    if (total == 0) {
        return 0.0f;
    }
    return static_cast<float>(correct_) / static_cast<float>(total);
}

void CCELoss::reset_score() {
    loss_ = 0.0f;
    cumulative_loss_ = 0.0f;
    correct_ = 0;
    incorrect_ = 0;
}