#include "DistributedDataLoader.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include "SGDOptimizer.hpp"
#include "Stream.hpp"
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

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

Model create_model(int rank, unsigned int seed = 0) {
    Model model(input_dim, hidden_dim, output_dim, batch_size);
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

    Model model = create_model(rank, 42);
    SGDOptimizer optimizer(0.3f);

    const int local_batch_size = batch_size / world_size;

    std::vector<float> host_images, host_labels;

    Tensor batch_x(local_batch_size, input_dim);
    Tensor batch_y(local_batch_size, output_dim);

    DistributedDataLoader data_loader(train_data, rank, world_size, batch_size,
                                      42);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        int batch_count = 0;

        while (data_loader.next_batch(host_images, host_labels)) {
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);

            float loss = model.forward(batch_x, batch_y);
            model.backward();

            // all reduce gradients across all ranks before optimizer step
            NCCLCHECK(ncclAllReduce(model.layer1().weight_grads().data(),
                                    model.layer1().weight_grads().data(),
                                    model.layer1().weight_grads().size(),
                                    ncclFloat, ncclSum, comm,
                                    get_cuda_stream()));
            NCCLCHECK(ncclAllReduce(model.layer1().bias_grads().data(),
                                    model.layer1().bias_grads().data(),
                                    model.layer1().bias_grads().size(),
                                    ncclFloat, ncclSum, comm,
                                    get_cuda_stream()));
            NCCLCHECK(ncclAllReduce(model.layer2().weight_grads().data(),
                                    model.layer2().weight_grads().data(),
                                    model.layer2().weight_grads().size(),
                                    ncclFloat, ncclSum, comm,
                                    get_cuda_stream()));
            NCCLCHECK(ncclAllReduce(model.layer2().bias_grads().data(),
                                    model.layer2().bias_grads().data(),
                                    model.layer2().bias_grads().size(),
                                    ncclFloat, ncclSum, comm,
                                    get_cuda_stream()));

            // Make sure all NCCL all-reduces finish first
            CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));

            model.layer1().weight_grads().div(world_size);
            model.layer1().bias_grads().div(world_size);
            model.layer2().weight_grads().div(world_size);
            model.layer2().bias_grads().div(world_size);

            optimizer.step(model.layer2());
            optimizer.step(model.layer1());

            ++batch_count;

            if (rank == 0 && batch_count % 100 == 0) {
                std::printf("  Batch %d | loss = %.6f | acc = %.4f\n",
                            batch_count, loss, model.accuracy());
            }
        }

        if (rank == 0) {
            std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                        epoch + 1, model.avg_loss(), model.accuracy());
        }
    }

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
    CUDACHECK(cudaSetDevice(0));
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