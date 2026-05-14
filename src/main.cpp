#include "MNIST.hpp"
#include "Model.hpp"
#include "SGDOptimizer.hpp"
#include "Stream.hpp"
#include "Tensor.hpp"

#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int batch_size = 80;
static constexpr int input_dim = 784;
static constexpr int hidden_dim = 32;
static constexpr int output_dim = 10;
static constexpr int num_epochs = 5;

Model create_model() {
    Model model(input_dim, hidden_dim, output_dim, batch_size);
    model.init();
    return model;
}

void train(const char* data_dir, const char* output_path) {
    std::printf("Executing training routine\n");

    std::string base(data_dir);

    MNIST train_data(
        base + "/train-images-idx3-ubyte",
        base + "/train-labels-idx1-ubyte"
    );

    Model model = create_model();
    SGDOptimizer optimizer(0.3f);

    Tensor batch_x(batch_size, input_dim);
    Tensor batch_y(batch_size, output_dim);

    std::vector<float> host_images;
    std::vector<float> host_labels;

    cudaEvent_t iter_start, compute_end;
    cudaEventCreate(&iter_start);
    cudaEventCreate(&compute_end);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::printf("Epoch %d\n", epoch + 1);

        train_data.reset();
        model.reset_score();

        int batch_count = 0;
        float epoch_compute_ms = 0.0f;

        while (train_data.next_batch(batch_size, host_images, host_labels)) {
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);

            cudaEventRecord(iter_start, get_cuda_stream());

            float loss = model.forward(batch_x, batch_y);
            model.backward();

            cudaEventRecord(compute_end, get_cuda_stream());
            cudaStreamSynchronize(get_cuda_stream());

            float compute_ms;
            cudaEventElapsedTime(&compute_ms, iter_start, compute_end);
            epoch_compute_ms += compute_ms;

            optimizer.step(model.layer2());
            optimizer.step(model.layer1());

            ++batch_count;

            if (batch_count % 100 == 0) {
                std::printf("  Batch %d | loss = %.6f | acc = %.4f\n",
                            batch_count, loss, model.accuracy());
            }
        }

        std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                    epoch + 1, model.avg_loss(), model.accuracy());
        std::printf("  compute = %.1f ms\n", epoch_compute_ms);
    }

    cudaEventDestroy(iter_start);
    cudaEventDestroy(compute_end);

    model.save(output_path);
    std::printf("Saved model to %s\n", output_path);
}

void evaluate(const char* data_dir, const char* model_path) {
    std::printf("Executing evaluation routine\n");

    std::string base(data_dir);

    MNIST test_data(
        base + "/t10k-images-idx3-ubyte",
        base + "/t10k-labels-idx1-ubyte"
    );

    Model model(input_dim, hidden_dim, output_dim, batch_size);
    model.load(model_path);
    model.reset_score();

    Tensor batch_x(batch_size, input_dim);
    Tensor batch_y(batch_size, output_dim);

    std::vector<float> host_images;
    std::vector<float> host_labels;

    int batch_count = 0;

    while (test_data.next_batch(batch_size, host_images, host_labels)) {
        batch_x.copy_from_host(host_images);
        batch_y.copy_from_host(host_labels);

        model.forward(batch_x, batch_y);
        ++batch_count;
    }

    std::printf("Evaluation complete over %d batches\n", batch_count);
    std::printf("Test avg loss = %.6f | test acc = %.4f\n",
                model.avg_loss(), model.accuracy());
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::printf("Usage:\n");
        std::printf("  %s train <data_dir> [output_model]\n", argv[0]);
        std::printf("  %s evaluate <data_dir> <model_path>\n", argv[0]);
        return 1;
    }

    try {
        if (std::strcmp(argv[1], "train") == 0) {
            const char* data_dir = argv[2];
            const char* output_path = (argc >= 4) ? argv[3] : "ff.params";
            train(data_dir, output_path);
        } else if (std::strcmp(argv[1], "evaluate") == 0) {
            if (argc < 4) {
                std::printf("evaluate requires <data_dir> and <model_path>\n");
                return 1;
            }
            evaluate(argv[2], argv[3]);
        } else {
            std::printf("Unrecognized command: %s\n", argv[1]);
            return 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }

    return 0;
}