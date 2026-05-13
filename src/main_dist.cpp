#include "CIFAR10.hpp"
#include "DistributedDataLoader.hpp"
#include "FFLayer.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include "SGDOptimizer.hpp"
#include "Stream.hpp"
#include <algorithm>
#include <cstring>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

#define MPICHECK(cmd)                                                        \
    do                                                                       \
    {                                                                        \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS)                                                \
        {                                                                    \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess)                                             \
        {                                                                 \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NCCLCHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess)                                             \
        {                                                                 \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

static constexpr int batch_size = 1600;
static constexpr int num_epochs = 10;

Model create_model(int rank, int local_batch_size, int input_dim, int output_dim,
                   int hidden_dim, unsigned int seed = 0)
{
    Model model(input_dim, hidden_dim, output_dim, local_batch_size);
    if (rank == 0)
        model.init(seed);

    MPICHECK(MPI_Bcast(model.layer1().weights().data(),
                       model.layer1().weights().size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer1().biases().data(),
                       model.layer1().biases().size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer2().weights().data(),
                       model.layer2().weights().size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer2().biases().data(),
                       model.layer2().biases().size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer3().weights().data(),
                       model.layer3().weights().size(), MPI_FLOAT, 0, MPI_COMM_WORLD));
    MPICHECK(MPI_Bcast(model.layer3().biases().data(),
                       model.layer3().biases().size(), MPI_FLOAT, 0, MPI_COMM_WORLD));

    return model;
}

template<typename Dataset>
void train(int rank, int world_size, Dataset& train_data,
           const char *output_path, ncclComm_t comm, int input_dim, int output_dim,
           int hidden_dim, int bucket_size, int accum_steps, const char *algo,
           float learning_rate)
{
    std::printf("Executing training routine\n");
    if (rank == 0)
        std::printf("NCCL algorithm: %s | accum_steps: %d | input_dim: %d | hidden_dim: %d | lr: %.4f\n",
                    algo, accum_steps, input_dim, hidden_dim, learning_rate);

    SGDOptimizer optimizer(learning_rate);

    const int local_batch_size = batch_size / world_size;
    Model model = create_model(rank, local_batch_size, input_dim, output_dim, hidden_dim, 42);

    std::vector<float> host_images, host_labels;

    Tensor batch_x(local_batch_size, input_dim);
    Tensor batch_y(local_batch_size, output_dim);

    DistributedDataLoader data_loader(train_data, rank, world_size, batch_size, 42);

    std::vector<Tensor *> all_grads = {
        &model.layer3().weight_grads(), &model.layer3().bias_grads(),
        &model.layer2().weight_grads(), &model.layer2().bias_grads(),
        &model.layer1().weight_grads(), &model.layer1().bias_grads()
    };

    int total_elems = 0;
    for (Tensor *t : all_grads) total_elems += static_cast<int>(t->size());

    if (bucket_size <= 0)
        bucket_size = total_elems;

    Tensor grad_buf(1, total_elems), accum_buf(1, total_elems);

    if (rank == 0)
        std::printf("Total grad elems: %d | bucket_size: %d (%d buckets)\n",
                    total_elems, bucket_size,
                    (total_elems + bucket_size - 1) / bucket_size);

    cudaEvent_t iter_start, compute_end, comm_start, comm_end;
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&comm_start));
    CUDACHECK(cudaEventCreate(&comm_end));

    auto pack = [&]()
    {
        int off = 0;
        for (Tensor *t : all_grads)
        {
            CUDACHECK(cudaMemcpyAsync(grad_buf.data() + off, t->data(),
                                      t->size() * sizeof(float),
                                      cudaMemcpyDeviceToDevice, get_cuda_stream()));
            off += static_cast<int>(t->size());
        }
    };

    auto unpack = [&]()
    {
        int off = 0;
        for (Tensor *t : all_grads)
        {
            CUDACHECK(cudaMemcpyAsync(t->data(), accum_buf.data() + off,
                                      t->size() * sizeof(float),
                                      cudaMemcpyDeviceToDevice, get_cuda_stream()));
            off += static_cast<int>(t->size());
        }
    };

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        if (rank == 0)
            std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        CUDACHECK(cudaMemsetAsync(accum_buf.data(), 0,
                                  accum_buf.bytes(), get_cuda_stream()));

        int batch_count = 0;
        int accum_count = 0;
        float epoch_compute_ms = 0.0f;
        float epoch_comm_ms = 0.0f;

        while (data_loader.next_batch(host_images, host_labels))
        {
            nvtxRangePushA("iter");

            nvtxRangePushA("h2d");
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);
            nvtxRangePop();

            CUDACHECK(cudaEventRecord(iter_start, get_cuda_stream()));

            nvtxRangePushA("fwd");
            float loss = model.forward(batch_x, batch_y);
            nvtxRangePop();

            nvtxRangePushA("bwd");
            model.backward();
            CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));
            nvtxRangePop();

            nvtxRangePushA("pack");
            pack();
            accum_buf.accumulate(grad_buf, get_cuda_stream());
            CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));
            nvtxRangePop();

            CUDACHECK(cudaEventRecord(compute_end, get_cuda_stream()));

            ++accum_count;

            if (accum_count == accum_steps)
            {
                CUDACHECK(cudaEventRecord(comm_start, get_cuda_stream()));

                nvtxRangePushA("allreduce");
                int off = 0;
                while (off < total_elems)
                {
                    int count = std::min(bucket_size, total_elems - off);
                    NCCLCHECK(ncclAllReduce(accum_buf.data() + off,
                                            accum_buf.data() + off,
                                            count, ncclFloat, ncclSum, comm,
                                            get_cuda_stream()));
                    off += count;
                }
                CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));
                nvtxRangePop();

                CUDACHECK(cudaEventRecord(comm_end, get_cuda_stream()));

                float compute_ms, comm_ms;
                CUDACHECK(cudaEventElapsedTime(&compute_ms, iter_start, compute_end));
                CUDACHECK(cudaEventElapsedTime(&comm_ms, comm_start, comm_end));
                epoch_compute_ms += compute_ms;
                epoch_comm_ms += comm_ms;

                const float scale = static_cast<float>(world_size * accum_steps);
                nvtxRangePushA("unpack");
                unpack();
                CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));
                nvtxRangePop();

                nvtxRangePushA("opt");
                for (FFLayer *layer : model.layers())
                {
                    layer->weight_grads().div(scale);
                    layer->bias_grads().div(scale);
                }

                optimizer.step(model.layer3());
                optimizer.step(model.layer2());
                optimizer.step(model.layer1());
                CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));
                nvtxRangePop();

                CUDACHECK(cudaMemsetAsync(accum_buf.data(), 0,
                                          accum_buf.bytes(), get_cuda_stream()));
                accum_count = 0;
            }

            ++batch_count;

            CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));
            nvtxRangePop();

            if (rank == 0 && batch_count % 100 == 0)
            {
                std::printf("  Batch %d | loss = %.6f | acc = %.4f\n",
                            batch_count, loss, model.accuracy());
            }
        }

        if (rank == 0)
        {
            std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                        epoch + 1, model.avg_loss(), model.accuracy());
            std::printf("  compute = %.1f ms | comm = %.1f ms | total = %.1f ms | comm %% = %.1f%%\n",
                        epoch_compute_ms, epoch_comm_ms,
                        epoch_compute_ms + epoch_comm_ms,
                        100.0f * epoch_comm_ms / (epoch_compute_ms + epoch_comm_ms));
        }
    }

    CUDACHECK(cudaEventDestroy(iter_start));
    CUDACHECK(cudaEventDestroy(compute_end));
    CUDACHECK(cudaEventDestroy(comm_start));
    CUDACHECK(cudaEventDestroy(comm_end));

    if (rank == 0)
    {
        model.save(output_path);
        std::printf("Saved model to %s\n", output_path);
    }
}

int main(int argc, char **argv)
{
    int rank, size;

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    ncclUniqueId id;
    if (rank == 0)
        NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    // argv[1]: dataset     ("mnist" default | "cifar10")
    // argv[2]: hidden_dim  (default 1024 for mnist, 4096 for cifar10)
    // argv[3]: bucket_size (default -1 = one bucket)
    // argv[4]: accum_steps (default 1 = no accumulation)
    // argv[5]: algo        ("ring", "tree", or omit to let NCCL decide)
    // argv[6]: learning_rate (default 0.3 for mnist, 0.01 for cifar10)
    const char *dataset_arg = (argc >= 2) ? argv[1] : "mnist";
    bool use_cifar10 = (std::strcmp(dataset_arg, "cifar10") == 0);
    int default_hidden = use_cifar10 ? 4096 : 1024;
    int hidden_dim  = (argc >= 3) ? std::atoi(argv[2]) : default_hidden;
    int bucket_size = (argc >= 4) ? std::atoi(argv[3]) : -1;
    int accum_steps = (argc >= 5) ? std::atoi(argv[4]) : 1;
    const char *algo_arg = (argc >= 6) ? argv[5] : "auto";
    float default_lr = use_cifar10 ? 0.01f : 0.3f;
    float learning_rate = (argc >= 7) ? std::atof(argv[6]) : default_lr;

    if (accum_steps < 1)
        accum_steps = 1;

    // Must be set before ncclCommInitRank so NCCL picks up the algorithm.
    if (std::strcmp(algo_arg, "ring") == 0)
        setenv("NCCL_ALGO", "Ring", 1);
    else if (std::strcmp(algo_arg, "tree") == 0)
        setenv("NCCL_ALGO", "Tree", 1);

    int local_rank = 0;
    const char *local_rank_env = std::getenv("SLURM_LOCALID");
    if (local_rank_env != nullptr)
        local_rank = std::atoi(local_rank_env);

    CUDACHECK(cudaSetDevice(local_rank));
    (void)get_cuda_stream();

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    const char data_dir[] = "data";
    const char output_path[] = "ff.params";

    if (use_cifar10) {
        CIFAR10 train_data(std::string(data_dir) + "/cifar-10-batches-bin", true);
        train(rank, size, train_data, output_path, comm,
              CIFAR10::IMAGE_DIM, CIFAR10::NUM_CLASSES,
              hidden_dim, bucket_size, accum_steps, algo_arg, learning_rate);
    } else {
        MNIST train_data(std::string(data_dir) + "/train-images-idx3-ubyte",
                         std::string(data_dir) + "/train-labels-idx1-ubyte");
        train(rank, size, train_data, output_path, comm,
              MNIST::IMAGE_DIM, MNIST::NUM_CLASSES,
              hidden_dim, bucket_size, accum_steps, algo_arg, learning_rate);
    }

    ncclCommDestroy(comm);
    MPICHECK(MPI_Finalize());
    return 0;
}
