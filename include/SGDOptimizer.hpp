#pragma once

#include "FFLayer.hpp"

#include <driver_types.h>

class SGDOptimizer {
public:
    explicit SGDOptimizer(float learning_rate);

    void step(FFLayer& layer);
    void step(FFLayer& layer, cudaStream_t stream);
    void step(FFLayer& layer, Tensor& weight_grad, Tensor& bias_grad, cudaStream_t stream);

    float learning_rate() const noexcept;

private:
    float learning_rate_;
};
