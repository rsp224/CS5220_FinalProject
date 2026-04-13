#pragma once

#include "Tensor.hpp"

#include <cstddef>
#include <vector>

class CCELoss {
public:
    CCELoss(int num_classes, int batch_size);

    // logits: [batch_size, num_classes]
    // target: [batch_size, num_classes] one-hot encoded
    //
    // Returns average batch loss.
    float forward(const Tensor& logits, const Tensor& target);

    // grad_logits: [batch_size, num_classes]
    // Fills with dL/dlogits from the most recent forward pass.
    void backward(Tensor& grad_logits);

    float avg_loss() const noexcept;
    float accuracy() const noexcept;
    void reset_score();

private:
    int num_classes_;
    int batch_size_;
    float inv_batch_size_;

    // Cached from last forward pass
    Tensor probs_;    // softmax probabilities, shape [batch_size, num_classes]
    Tensor target_;   // cached target, same shape

    float loss_;
    float cumulative_loss_;
    std::size_t correct_;
    std::size_t incorrect_;
};