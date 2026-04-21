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

    void save(const std::string& filename) const;
    void load(const std::string& filename);

    float avg_loss() const noexcept;
    float accuracy() const noexcept;
    void reset_score();

    FFLayer& layer1() noexcept;
    FFLayer& layer2() noexcept;
    std::vector<FFLayer*> layers() noexcept;

private:
    int input_dim_;
    int hidden_dim_;
    int output_dim_;
    int batch_size_;

    FFLayer layer1_;
    FFLayer layer2_;
    CCELoss loss_;

    Tensor hidden_;
    Tensor logits_;
    Tensor grad_logits_;
    Tensor grad_hidden_;
    Tensor grad_input_;
};