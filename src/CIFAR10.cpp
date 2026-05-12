#include "CIFAR10.hpp"

#include <cstdio>
#include <stdexcept>

static constexpr int RECORDS_PER_FILE = 10000;
static constexpr int RECORD_BYTES     = 1 + CIFAR10::IMAGE_DIM; // label + pixels

CIFAR10::CIFAR10(const std::string& data_dir, bool train)
    : current_file_(0), total_samples_(0), current_index_(0), file_offset_(0) {
    if (train) {
        for (int i = 1; i <= 5; ++i) {
            batch_files_.push_back(data_dir + "/data_batch_" + std::to_string(i) + ".bin");
        }
        total_samples_ = 5 * RECORDS_PER_FILE;
    } else {
        batch_files_.push_back(data_dir + "/test_batch.bin");
        total_samples_ = RECORDS_PER_FILE;
    }
    open_file(0);
    std::printf("Loaded CIFAR-10 (%s) with %zu samples\n",
                train ? "train" : "test", total_samples_);
}

void CIFAR10::open_file(int idx) {
    stream_.close();
    stream_.open(batch_files_[idx], std::ios::binary);
    if (!stream_.is_open()) {
        throw std::runtime_error("Could not open CIFAR-10 file: " + batch_files_[idx]);
    }
    file_offset_ = 0;
}

std::size_t CIFAR10::size() const noexcept {
    return total_samples_;
}

void CIFAR10::reset() {
    current_index_ = 0;
    current_file_  = 0;
    open_file(0);
}

bool CIFAR10::read_one(std::vector<float>& image_out, std::vector<float>& label_out) {
    if (current_index_ >= total_samples_) {
        return false;
    }

    // Advance to next file when current one is exhausted.
    if (file_offset_ >= RECORDS_PER_FILE) {
        ++current_file_;
        if (current_file_ >= static_cast<int>(batch_files_.size())) {
            return false;
        }
        open_file(current_file_);
    }

    char buf[RECORD_BYTES];
    stream_.read(buf, RECORD_BYTES);
    if (!stream_) {
        return false;
    }

    unsigned char label = static_cast<unsigned char>(buf[0]);
    if (label >= NUM_CLASSES) {
        throw std::runtime_error("Invalid CIFAR-10 label encountered.");
    }

    label_out.assign(NUM_CLASSES, 0.0f);
    label_out[label] = 1.0f;

    image_out.resize(IMAGE_DIM);
    constexpr float inv_255 = 1.0f / 255.0f;
    for (int i = 0; i < IMAGE_DIM; ++i) {
        image_out[i] = static_cast<unsigned char>(buf[1 + i]) * inv_255;
    }

    ++file_offset_;
    ++current_index_;
    return true;
}

bool CIFAR10::next_batch(int batch_size,
                         std::vector<float>& images_out,
                         std::vector<float>& labels_out) {
    if (batch_size <= 0) {
        throw std::runtime_error("Batch size must be positive.");
    }

    if (current_index_ + static_cast<std::size_t>(batch_size) > total_samples_) {
        return false;
    }

    images_out.resize(static_cast<std::size_t>(batch_size) * IMAGE_DIM);
    labels_out.resize(static_cast<std::size_t>(batch_size) * NUM_CLASSES);

    std::vector<float> one_image, one_label;
    for (int b = 0; b < batch_size; ++b) {
        if (!read_one(one_image, one_label)) {
            return false;
        }
        std::copy(one_image.begin(), one_image.end(),
                  images_out.begin() + static_cast<std::size_t>(b) * IMAGE_DIM);
        std::copy(one_label.begin(), one_label.end(),
                  labels_out.begin() + static_cast<std::size_t>(b) * NUM_CLASSES);
    }
    return true;
}
