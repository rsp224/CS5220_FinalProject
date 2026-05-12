#include "DistributedDataLoader.hpp"

void DistributedDataLoader::reset() {
    current_index_ = 0;

    indices_.resize(num_samples_);
    std::iota(indices_.begin(), indices_.end(), 0);
    std::shuffle(indices_.begin(), indices_.end(), rng_);

    const std::size_t usable = (num_samples_ / static_cast<std::size_t>(batch_size_))
                               * static_cast<std::size_t>(batch_size_);
    indices_.resize(usable);
}

bool DistributedDataLoader::next_batch(std::vector<float>& batch_images,
                                       std::vector<float>& batch_labels) {
    const int local_batch_size = batch_size_ / world_size_;
    const int shard_len  = static_cast<int>(indices_.size()) / world_size_;
    const int shard_base = rank_ * shard_len;

    if (current_index_ + local_batch_size > shard_len) {
        return false;
    }

    const int shard_start = shard_base + current_index_;

    batch_images.clear();
    batch_labels.clear();
    batch_images.reserve(static_cast<std::size_t>(local_batch_size) * image_dim_);
    batch_labels.reserve(static_cast<std::size_t>(local_batch_size) * num_classes_);

    for (int i = 0; i < local_batch_size; ++i) {
        int idx = indices_[shard_start + i];
        batch_images.insert(batch_images.end(),
                            images_.begin() + idx * image_dim_,
                            images_.begin() + (idx + 1) * image_dim_);
        batch_labels.insert(batch_labels.end(),
                            labels_.begin() + idx * num_classes_,
                            labels_.begin() + (idx + 1) * num_classes_);
    }

    current_index_ += local_batch_size;
    return true;
}
