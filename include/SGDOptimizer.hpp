#pragma once

#include "FFLayer.hpp"

class SGDOptimizer {
public:
    explicit SGDOptimizer(float learning_rate);

    void step(FFLayer& layer);

    float learning_rate() const noexcept;

private:
    float learning_rate_;
};