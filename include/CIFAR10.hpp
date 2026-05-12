#pragma once

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>

// Loads CIFAR-10 from the official binary batch files
// (data_batch_1.bin ... data_batch_5.bin for train, test_batch.bin for test).
// Each record: 1-byte label + 3072 bytes (1024 R, 1024 G, 1024 B), row-major.
class CIFAR10 {
public:
    static constexpr int IMAGE_ROWS     = 32;
    static constexpr int IMAGE_COLS     = 32;
    static constexpr int IMAGE_CHANNELS = 3;
    static constexpr int IMAGE_DIM      = IMAGE_ROWS * IMAGE_COLS * IMAGE_CHANNELS; // 3072
    static constexpr int NUM_CLASSES    = 10;

    // data_dir: path containing data_batch_{1-5}.bin and test_batch.bin
    // train: true = load all 5 training batches (50k samples)
    //        false = load test_batch.bin (10k samples)
    CIFAR10(const std::string& data_dir, bool train);

    std::size_t size() const noexcept;
    void reset();

    bool next_batch(int batch_size,
                    std::vector<float>& images_out,
                    std::vector<float>& labels_out);

    bool read_one(std::vector<float>& image_out, std::vector<float>& label_out);

private:
    std::vector<std::string> batch_files_;
    int current_file_;
    std::ifstream stream_;
    std::size_t total_samples_;
    std::size_t current_index_;
    int file_offset_;  // records consumed from current open file

    void open_file(int idx);
};
