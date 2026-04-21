#include "DistributedDataLoader.hpp"
#include "FFLayer.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include "SGDOptimizer.hpp"
#include "Stream.hpp"
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>

#define MPICHECK(cmd)                                                          \
    do {                                                                       \
        int e = cmd;                                                           \
        if (e != MPI_SUCCESS) {                                                \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);   \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define NCCLCHECK(cmd)                                                         \
    do {                                                                       \
        ncclResult_t r = cmd;                                                  \
        if (r != ncclSuccess) {                                                \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,      \
                   ncclGetErrorString(r));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static constexpr int batch_size = 80;
static constexpr int input_dim = 784;
static constexpr int hidden_dim = 32;
static constexpr int output_dim = 10;
static constexpr int num_epochs = 5;

Model create_model(int rank, int local_batch_size, unsigned int seed = 0) {
    Model model(input_dim, hidden_dim, output_dim, local_batch_size);
    // only initialize on rank 0 and broadcast model params to others
    if (rank == 0) {
        model.init(seed);
    }

    MPICHECK(MPI_Bcast(model.layer1().weights().data(),
                       model.layer1().weights().size(), MPI_FLOAT, 0,
                       MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer1().biases().data(),
                       model.layer1().biases().size(), MPI_FLOAT, 0,
                       MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer2().weights().data(),
                       model.layer2().weights().size(), MPI_FLOAT, 0,
                       MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer2().biases().data(),
                       model.layer2().biases().size(), MPI_FLOAT, 0,
                       MPI_COMM_WORLD));

    return model;
}

void train(int rank, int world_size, const char *data_dir,
           const char *output_path, ncclComm_t comm) {
    std::printf("Executing training routine\n");

    std::string base(data_dir);

    MNIST train_data(base + "/train-images-idx3-ubyte",
                     base + "/train-labels-idx1-ubyte");

    SGDOptimizer optimizer(0.3f);

    const int local_batch_size = batch_size / world_size;
    Model model = create_model(rank, local_batch_size, 42);

    std::vector<float> host_images, host_labels;

    Tensor batch_x(local_batch_size, input_dim);
    Tensor batch_y(local_batch_size, output_dim);

    DistributedDataLoader data_loader(train_data, rank, world_size, batch_size,
                                      42);

    cudaEvent_t iter_start, compute_end, comm_end;
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&comm_end));

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        if (rank == 0)
            std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        int batch_count = 0;
        float epoch_compute_ms = 0.0f;
        float epoch_comm_ms = 0.0f;

        while (data_loader.next_batch(host_images, host_labels)) {
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);

            CUDACHECK(cudaEventRecord(iter_start, get_cuda_stream()));

            float loss = model.forward(batch_x, batch_y);
            model.backward();

            CUDACHECK(cudaEventRecord(compute_end, get_cuda_stream()));

            // all reduce gradients across all ranks before optimizer step
            for (FFLayer* layer : model.layers()) {
                NCCLCHECK(ncclAllReduce(layer->weight_grads().data(),
                                        layer->weight_grads().data(),
                                        layer->weight_grads().size(),
                                        ncclFloat, ncclSum, comm,
                                        get_cuda_stream()));
                NCCLCHECK(ncclAllReduce(layer->bias_grads().data(),
                                        layer->bias_grads().data(),
                                        layer->bias_grads().size(),
                                        ncclFloat, ncclSum, comm,
                                        get_cuda_stream()));
            }

            CUDACHECK(cudaEventRecord(comm_end, get_cuda_stream()));

            // Sync stream — all three events are now complete after this
            CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));

            float compute_ms, comm_ms;
            CUDACHECK(cudaEventElapsedTime(&compute_ms, iter_start, compute_end));
            CUDACHECK(cudaEventElapsedTime(&comm_ms, compute_end, comm_end));
            epoch_compute_ms += compute_ms;
            epoch_comm_ms += comm_ms;

            for (FFLayer* layer : model.layers()) {
                layer->weight_grads().div(world_size);
                layer->bias_grads().div(world_size);
            }

            optimizer.step(model.layer2());
            optimizer.step(model.layer1());

            ++batch_count;

            if (rank == 0 && batch_count % 100 == 0) {
                std::printf("  Batch %d | loss = %.6f | acc = %.4f\n",
                            batch_count, loss, model.accuracy());
            }
        }

        if (rank == 0) {
            float total_ms = epoch_compute_ms + epoch_comm_ms;
            float comm_pct = 100.0f * epoch_comm_ms / total_ms;
            std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                        epoch + 1, model.avg_loss(), model.accuracy());
            std::printf("  compute = %.1f ms | comm = %.1f ms | total = %.1f ms | comm %% = %.1f%%\n",
                        epoch_compute_ms, epoch_comm_ms, total_ms, comm_pct);
        }
    }

    CUDACHECK(cudaEventDestroy(iter_start));
    CUDACHECK(cudaEventDestroy(compute_end));
    CUDACHECK(cudaEventDestroy(comm_end));

    if (rank == 0) {
        model.save(output_path);
        std::printf("Saved model to %s\n", output_path);
    }
}

int main(int argc, char **argv) {
    int rank, size;

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    ncclUniqueId id;
    if (rank == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // Slurm should already handle GPU assignment where it assigns one GPU per
    // rank
    int local_rank = 0;
    const char* local_rank_env = std::getenv("SLURM_LOCALID");
    if (local_rank_env != nullptr) {
        local_rank = std::atoi(local_rank_env);
    }
    CUDACHECK(cudaSetDevice(local_rank));
    // Warm up the lazy stream singleton so NCCL and kernels share it.
    (void)get_cuda_stream();

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    const char data_dir[] = "data";
    const char output_path[] = "ff.params";

    train(rank, size, data_dir, output_path, comm);

    ncclCommDestroy(comm);
    // Stream is owned by the get_cuda_stream() static and is released at exit.
    MPICHECK(MPI_Finalize());
    return 0;
}