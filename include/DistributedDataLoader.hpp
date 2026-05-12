#pragma once

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>

class DistributedDataLoader {
  public:
    // Works with any dataset type that exposes:
    //   static constexpr int IMAGE_DIM, NUM_CLASSES
    //   bool read_one(vector<float>&, vector<float>&)
    template<typename Dataset>
    DistributedDataLoader(Dataset& dataset, int rank, int world_size,
                          int batch_size, int seed)
        : rank_(rank), world_size_(world_size), batch_size_(batch_size),
          current_index_(0), image_dim_(Dataset::IMAGE_DIM),
          num_classes_(Dataset::NUM_CLASSES) {
        assert(batch_size % world_size == 0);

        std::vector<float> image, label;
        while (dataset.read_one(image, label)) {
            images_.insert(images_.end(), image.begin(), image.end());
            labels_.insert(labels_.end(), label.begin(), label.end());
        }

        assert(images_.size() % static_cast<std::size_t>(image_dim_) == 0);
        num_samples_ = images_.size() / static_cast<std::size_t>(image_dim_);
        assert(labels_.size() == num_samples_ * static_cast<std::size_t>(num_classes_));

        rng_ = std::mt19937(static_cast<unsigned>(seed));
    }

    void reset();
    bool next_batch(std::vector<float>& batch_images,
                    std::vector<float>& batch_labels);

  private:
    int rank_, world_size_, batch_size_, current_index_;
    int image_dim_, num_classes_;
    std::size_t num_samples_;
    std::vector<float> images_, labels_;
    std::vector<int> indices_;
    std::mt19937 rng_;
};
