#pragma once

#include "Tensor.hpp"

#include <string>

enum class Activation {
    None,
    ReLU
};

class FFLayer {
public:
    FFLayer(std::string name,
            Activation activation,
            int output_size,
            int input_size);

    void init(unsigned int seed);

    void forward(const Tensor& input, Tensor& output);
    void backward_parameter_grads(const Tensor& grad_output);
    void backward_input_grad(Tensor& grad_input);

    void zero_grad_accum(cudaStream_t stream);
    void accumulate_grads(cudaStream_t stream);

    Tensor& weights() noexcept { return W_; }
    Tensor& biases() noexcept { return b_; }
    Tensor& weight_grads() noexcept { return dW_; }
    Tensor& bias_grads() noexcept { return db_; }
    Tensor& weight_grad_accum() noexcept { return dW_accum_; }
    Tensor& bias_grad_accum() noexcept { return db_accum_; }

    const Tensor& weights() const noexcept { return W_; }
    const Tensor& biases() const noexcept { return b_; }
    const Tensor& weight_grads() const noexcept { return dW_; }
    const Tensor& bias_grads() const noexcept { return db_; }
    const Tensor& weight_grad_accum() const noexcept { return dW_accum_; }
    const Tensor& bias_grad_accum() const noexcept { return db_accum_; }

    int input_size() const noexcept { return input_size_; }
    int output_size() const noexcept { return output_size_; }

private:
    std::string name_;
    Activation activation_;
    int output_size_;
    int input_size_;

    Tensor W_;
    Tensor b_;

    Tensor dW_;
    Tensor db_;

    Tensor dW_accum_;
    Tensor db_accum_;

    Tensor input_cache_;
    Tensor preact_cache_;
    Tensor output_cache_;
    Tensor act_grad_cache_;
};
