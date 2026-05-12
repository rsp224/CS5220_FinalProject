#pragma once

#include "CCELoss.hpp"
#include "FFLayer.hpp"
#include "Tensor.hpp"

#include <string>
#include <vector>

class Model {
public:
    Model(int input_dim, int hidden_dim, int output_dim, int batch_size);

    void init(unsigned int seed = 0);

    float forward(const Tensor& input, const Tensor& target);
    void backward();
    void backward_upper();  // loss + layer3; produces layer3 grads and grad_hidden2_
    void backward_lower();  // layer2 + layer1; must follow backward_upper()

    void save(const std::string& filename) const;
    void load(const std::string& filename);

    float avg_loss() const noexcept;
    float accuracy() const noexcept;
    void reset_score();

    FFLayer& layer1() noexcept;
    FFLayer& layer2() noexcept;
    FFLayer& layer3() noexcept;
    std::vector<FFLayer*> layers() noexcept;

private:
    int input_dim_;
    int hidden_dim_;
    int output_dim_;
    int batch_size_;

    FFLayer layer1_;  // input_dim  → hidden_dim, ReLU
    FFLayer layer2_;  // hidden_dim → hidden_dim, ReLU
    FFLayer layer3_;  // hidden_dim → output_dim, None
    CCELoss loss_;

    Tensor hidden_;        // [batch, hidden_dim]
    Tensor hidden2_;       // [batch, hidden_dim]
    Tensor logits_;        // [batch, output_dim]
    Tensor grad_logits_;   // [batch, output_dim]
    Tensor grad_hidden2_;  // [batch, hidden_dim]
    Tensor grad_hidden_;   // [batch, hidden_dim]
    Tensor grad_input_;    // [batch, input_dim]
};
