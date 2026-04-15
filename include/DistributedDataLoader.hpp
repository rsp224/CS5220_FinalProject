#pragma once

#include "MNIST.hpp"
#include <random>
#include <vector>

class DistributedDataLoader {
  private:
    int rank_, world_size_, batch_size_, current_index_;
    size_t num_samples_;
    std::vector<float> images_, labels_;
    std::vector<int> indices_;
    std::mt19937 rng_;

  public:
    DistributedDataLoader(MNIST &dataset, int rank, int world_size,
                          int batch_size, int seed);
    // root node randomly shuffles indices, scatters to all
    void reset();
    // get local batch of images and labels, returns false if no more batches
    // available
    bool next_batch(std::vector<float> &batch_images,
                    std::vector<float> &batch_labels);
};