#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class MNIST {
public:
    static constexpr int IMAGE_ROWS = 28;
    static constexpr int IMAGE_COLS = 28;
    static constexpr int IMAGE_DIM  = IMAGE_ROWS * IMAGE_COLS;
    static constexpr int NUM_CLASSES = 10;

    MNIST(const std::string& image_path, const std::string& label_path);

    std::size_t size() const noexcept;
    void reset();

    // Reads the next batch from disk.
    // images_out will be resized to batch_size * IMAGE_DIM
    // labels_out will be resized to batch_size * NUM_CLASSES
    //
    // Returns false if there is not enough remaining data for a full batch.
    bool next_batch(int batch_size,
                    std::vector<float>& images_out,
                    std::vector<float>& labels_out);

    // Debug helper: prints the most recently read sample in ASCII
    void print_last() const;

private:
    void read_be_uint32(std::ifstream& in, uint32_t& out);
    void read_header();
    bool read_one(std::vector<float>& image_out, std::vector<float>& label_out);

    std::ifstream images_;
    std::ifstream labels_;

    std::size_t image_count_;
    std::size_t current_index_;

    std::vector<float> last_image_;
    std::vector<float> last_label_;
};