#include "MNIST.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

void MNIST::read_be_uint32(std::ifstream& in, uint32_t& out) {
    char* buf = reinterpret_cast<char*>(&out);
    in.read(buf, 4);

    if (!in) {
        throw std::runtime_error("Failed to read 4-byte integer from MNIST file.");
    }

    std::swap(buf[0], buf[3]);
    std::swap(buf[1], buf[2]);
}

MNIST::MNIST(const std::string& image_path, const std::string& label_path)
    : images_(image_path, std::ios::binary),
      labels_(label_path, std::ios::binary),
      image_count_(0),
      current_index_(0),
      last_image_(IMAGE_DIM, 0.0f),
      last_label_(NUM_CLASSES, 0.0f) {
    if (!images_.is_open()) {
        throw std::runtime_error("Could not open image file: " + image_path);
    }
    if (!labels_.is_open()) {
        throw std::runtime_error("Could not open label file: " + label_path);
    }

    read_header();
}

void MNIST::read_header() {
    uint32_t image_magic = 0;
    read_be_uint32(images_, image_magic);
    if (image_magic != 2051) {
        throw std::runtime_error("Images file appears to be malformed.");
    }

    uint32_t image_count_u32 = 0;
    read_be_uint32(images_, image_count_u32);
    image_count_ = static_cast<std::size_t>(image_count_u32);

    uint32_t label_magic = 0;
    read_be_uint32(labels_, label_magic);
    if (label_magic != 2049) {
        throw std::runtime_error("Labels file appears to be malformed.");
    }

    uint32_t label_count_u32 = 0;
    read_be_uint32(labels_, label_count_u32);
    if (static_cast<std::size_t>(label_count_u32) != image_count_) {
        throw std::runtime_error("Label count did not match image count.");
    }

    uint32_t rows = 0;
    uint32_t cols = 0;
    read_be_uint32(images_, rows);
    read_be_uint32(images_, cols);

    if (rows != IMAGE_ROWS || cols != IMAGE_COLS) {
        throw std::runtime_error("Expected 28x28 MNIST images.");
    }

    std::printf("Loaded MNIST with %zu samples\n", image_count_);
}

std::size_t MNIST::size() const noexcept {
    return image_count_;
}

void MNIST::reset() {
    images_.clear();
    labels_.clear();

    images_.seekg(16, std::ios::beg);  // image header = 16 bytes
    labels_.seekg(8, std::ios::beg);   // label header = 8 bytes

    current_index_ = 0;
}

bool MNIST::read_one(std::vector<float>& image_out, std::vector<float>& label_out) {
    if (current_index_ >= image_count_) {
        return false;
    }

    image_out.assign(IMAGE_DIM, 0.0f);
    label_out.assign(NUM_CLASSES, 0.0f);

    char buf[IMAGE_DIM];
    images_.read(buf, IMAGE_DIM);
    if (!images_) {
        return false;
    }

    constexpr float inv_255 = 1.0f / 255.0f;
    for (int i = 0; i < IMAGE_DIM; ++i) {
        image_out[i] = static_cast<unsigned char>(buf[i]) * inv_255;
    }

    char label_char = 0;
    labels_.read(&label_char, 1);
    if (!labels_) {
        return false;
    }

    unsigned char label = static_cast<unsigned char>(label_char);
    if (label >= NUM_CLASSES) {
        throw std::runtime_error("Invalid MNIST label encountered.");
    }

    label_out[label] = 1.0f;

    last_image_ = image_out;
    last_label_ = label_out;
    ++current_index_;

    return true;
}

bool MNIST::next_batch(int batch_size,
                       std::vector<float>& images_out,
                       std::vector<float>& labels_out) {
    if (batch_size <= 0) {
        throw std::runtime_error("Batch size must be positive.");
    }

    if (current_index_ + static_cast<std::size_t>(batch_size) > image_count_) {
        return false;
    }

    images_out.assign(static_cast<std::size_t>(batch_size) * IMAGE_DIM, 0.0f);
    labels_out.assign(static_cast<std::size_t>(batch_size) * NUM_CLASSES, 0.0f);

    std::vector<float> one_image;
    std::vector<float> one_label;

    for (int b = 0; b < batch_size; ++b) {
        if (!read_one(one_image, one_label)) {
            return false;
        }

        std::copy(
            one_image.begin(),
            one_image.end(),
            images_out.begin() + static_cast<std::size_t>(b) * IMAGE_DIM
        );

        std::copy(
            one_label.begin(),
            one_label.end(),
            labels_out.begin() + static_cast<std::size_t>(b) * NUM_CLASSES
        );
    }

    return true;
}

void MNIST::print_last() const {
    int label_index = -1;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        if (last_label_[i] == 1.0f) {
            label_index = i;
            break;
        }
    }

    if (label_index >= 0) {
        std::printf("Last sample label: %d\n", label_index);
    }

    for (int i = 0; i < IMAGE_ROWS; ++i) {
        int offset = i * IMAGE_COLS;
        for (int j = 0; j < IMAGE_COLS; ++j) {
            float v = last_image_[offset + j];
            if (v > 0.9f) {
                std::printf("#");
            } else if (v > 0.7f) {
                std::printf("*");
            } else if (v > 0.5f) {
                std::printf(".");
            } else {
                std::printf(" ");
            }
        }
        std::printf("\n");
    }
    std::printf("\n");
}