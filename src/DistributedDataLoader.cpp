#include "DistributedDataLoader.hpp"
#include "MNIST.hpp"
#include <algorithm>
#include <cassert>
#include <numeric>
#include <vector>

DistributedDataLoader::DistributedDataLoader(MNIST &dataset, int rank,
                                             int world_size, int batch_size,
                                             int seed)
    : rank_(rank), world_size_(world_size), batch_size_(batch_size),
      current_index_(0) {
    // For simplicity require the global batch to divide evenly across ranks.
    assert(batch_size % world_size == 0);

    std::vector<float> image;
    std::vector<float> label;
    while (dataset.read_one(image, label)) {
        images_.insert(images_.end(), image.begin(), image.end());
        labels_.insert(labels_.end(), label.begin(), label.end());
    }

    // Number of samples = total pixels / pixels-per-image.
    const int image_dim = MNIST::IMAGE_DIM;
    assert(images_.size() % image_dim == 0);
    num_samples_ = images_.size() / image_dim;
    assert(labels_.size() == num_samples_ * MNIST::NUM_CLASSES);

    rng_ = std::mt19937(seed);
}

void DistributedDataLoader::reset() {
    current_index_ = 0;

    // Rebuild the full index list every epoch so the discarded tail (samples
    // beyond the largest whole global batch) is a different random subset each
    // time — every sample gets a chance to be seen across epochs.
    indices_.resize(num_samples_);
    std::iota(indices_.begin(), indices_.end(), 0);

    // Deterministic shuffle — identical on every rank because all ranks were
    // seeded with the same value and consume the RNG in lockstep.
    std::shuffle(indices_.begin(), indices_.end(), rng_);

    // Trim to the largest whole number of global batches.
    const size_t usable = (num_samples_ / batch_size_) * batch_size_;
    indices_.resize(usable);
}

bool DistributedDataLoader::next_batch(std::vector<float> &batch_images,
                                       std::vector<float> &batch_labels) {
    const int local_batch_size = batch_size_ / world_size_;
    const int image_dim = MNIST::IMAGE_DIM;
    const int num_classes = MNIST::NUM_CLASSES;

    // Each rank owns a contiguous chunk of the shuffled indices:
    //   rank r -> indices_[r*shard_len : (r+1)*shard_len)
    // indices_ is trimmed to a multiple of batch_size_ in the constructor, and
    // batch_size_ is divisible by world_size_, so shard_len divides evenly.
    const int shard_len = (int)indices_.size() / world_size_;
    const int shard_base = rank_ * shard_len;

    // current_index_ is the local step offset inside this rank's shard.
    if (current_index_ + local_batch_size > shard_len) {
        return false;
    }

    const int shard_start = shard_base + current_index_;

    batch_images.clear();
    batch_labels.clear();
    batch_images.reserve(local_batch_size * image_dim);
    batch_labels.reserve(local_batch_size * num_classes);

    for (int i = 0; i < local_batch_size; ++i) {
        int idx = indices_[shard_start + i];
        batch_images.insert(batch_images.end(),
                            images_.begin() + idx * image_dim,
                            images_.begin() + (idx + 1) * image_dim);
        batch_labels.insert(batch_labels.end(),
                            labels_.begin() + idx * num_classes,
                            labels_.begin() + (idx + 1) * num_classes);
    }

    current_index_ += local_batch_size;
    return true;
}
