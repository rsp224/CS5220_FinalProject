#include "Model.hpp"

#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

Model::Model(int input_dim, int hidden_dim, int output_dim, int batch_size)
    : input_dim_(input_dim),
      hidden_dim_(hidden_dim),
      output_dim_(output_dim),
      batch_size_(batch_size),
      layer1_("layer1", Activation::ReLU, hidden_dim, input_dim),
      layer2_("layer2", Activation::ReLU, hidden_dim, hidden_dim),
      layer3_("layer3", Activation::None, output_dim, hidden_dim),
      loss_(output_dim, batch_size),
      hidden_(batch_size, hidden_dim),
      hidden2_(batch_size, hidden_dim),
      logits_(batch_size, output_dim),
      grad_logits_(batch_size, output_dim),
      grad_hidden2_(batch_size, hidden_dim),
      grad_hidden_(batch_size, hidden_dim) {}

void Model::init(unsigned int seed)
{
    layer1_.init(seed + 1);
    layer2_.init(seed + 2);
    layer3_.init(seed + 3);
}

float Model::forward(const Tensor& input, const Tensor& target)
{
    layer1_.forward(input, hidden_);
    layer2_.forward(hidden_, hidden2_);
    layer3_.forward(hidden2_, logits_);
    return loss_.forward(logits_, target);
}

void Model::backward()
{
    loss_.backward(grad_logits_);
    layer3_.backward(grad_logits_, grad_hidden2_);
    layer2_.backward(grad_hidden2_, grad_hidden_);
    layer1_.backward(grad_hidden_, grad_input_);
}

void Model::backward_upper()
{
    loss_.backward(grad_logits_);
    layer3_.backward(grad_logits_, grad_hidden2_);
}

void Model::backward_lower()
{
    layer2_.backward(grad_hidden2_, grad_hidden_);
    layer1_.backward(grad_hidden_, grad_input_);
}


float Model::avg_loss() const noexcept
{
    return loss_.avg_loss();
}

float Model::accuracy() const noexcept
{
    return loss_.accuracy();
}

void Model::reset_score()
{
    loss_.reset_score();
}

FFLayer& Model::layer1() noexcept { return layer1_; }
FFLayer& Model::layer2() noexcept { return layer2_; }
FFLayer& Model::layer3() noexcept { return layer3_; }

std::vector<FFLayer*> Model::layers() noexcept
{
    return {&layer1_, &layer2_, &layer3_};
}

void Model::save(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Failed to open file for saving: " + filename);

    auto write_tensor = [&out](const Tensor& t) {
        std::vector<float> host = t.copy_to_host();
        out.write(reinterpret_cast<const char*>(host.data()),
                  static_cast<std::streamsize>(host.size() * sizeof(float)));
    };

    write_tensor(layer1_.weights());
    write_tensor(layer1_.biases());
    write_tensor(layer2_.weights());
    write_tensor(layer2_.biases());
    write_tensor(layer3_.weights());
    write_tensor(layer3_.biases());
}

void Model::load(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Failed to open file for loading: " + filename);

    auto read_tensor = [&in](Tensor& t) {
        std::vector<float> host(t.size());
        in.read(reinterpret_cast<char*>(host.data()),
                static_cast<std::streamsize>(host.size() * sizeof(float)));
        if (!in)
            throw std::runtime_error("Failed while reading model file.");
        t.copy_from_host(host);
    };

    read_tensor(layer1_.weights());
    read_tensor(layer1_.biases());
    read_tensor(layer2_.weights());
    read_tensor(layer2_.biases());
    read_tensor(layer3_.weights());
    read_tensor(layer3_.biases());
}
