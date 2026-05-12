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

    // upper: layer3 grads (ready after backward_upper).
    // lower: layer2 + layer1 grads (ready after backward_lower).
    // On the final accumulation step, upper AllReduce runs on comm stream
    // concurrently with lower backward on compute stream.
    std::vector<Tensor *> upper_grads = {&model.layer3().weight_grads(),
                                         &model.layer3().bias_grads()};
    std::vector<Tensor *> lower_grads = {&model.layer2().weight_grads(),
                                         &model.layer2().bias_grads(),
                                         &model.layer1().weight_grads(),
                                         &model.layer1().bias_grads()};

    int upper_elems = 0, lower_elems = 0;
    for (Tensor *t : upper_grads) upper_elems += static_cast<int>(t->size());
    for (Tensor *t : lower_grads) lower_elems += static_cast<int>(t->size());

    if (bucket_size <= 0)
        bucket_size = upper_elems + lower_elems;

    Tensor grad_buf_upper(1, upper_elems), accum_buf_upper(1, upper_elems);
    Tensor grad_buf_lower(1, lower_elems), accum_buf_lower(1, lower_elems);

    if (rank == 0)
        std::printf("Grad elems: upper(layer3)=%d lower(layer2+1)=%d bucket_size=%d\n",
                    upper_elems, lower_elems, bucket_size);

    (void)get_comm_stream();

    // iter_start/compute_end on compute stream for timing.
    // layer3_ready: inter-stream signal after upper grads are accumulated.
    // comm_start/comm_end on comm stream for AllReduce timing.
    cudaEvent_t iter_start, layer3_ready, compute_end, comm_start, comm_end;
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&layer3_ready));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&comm_start));
    CUDACHECK(cudaEventCreate(&comm_end));

    auto pack = [&](Tensor &dst, std::vector<Tensor *> &srcs)
    {
        int off = 0;
        for (Tensor *t : srcs)
        {
            CUDACHECK(cudaMemcpyAsync(dst.data() + off, t->data(),
                                      t->size() * sizeof(float),
                                      cudaMemcpyDeviceToDevice, get_cuda_stream()));
            off += static_cast<int>(t->size());
        }
    };

    auto unpack = [&](std::vector<Tensor *> &dsts, Tensor &src)
    {
        int off = 0;
        for (Tensor *t : dsts)
        {
            CUDACHECK(cudaMemcpyAsync(t->data(), src.data() + off,
                                      t->size() * sizeof(float),
                                      cudaMemcpyDeviceToDevice, get_cuda_stream()));
            off += static_cast<int>(t->size());
        }
    };

    // AllReduce a flat buffer in bucket_size chunks on the comm stream.
    auto allreduce = [&](Tensor &buf, int elems)
    {
        int off = 0;
        while (off < elems)
        {
            int count = std::min(bucket_size, elems - off);
            NCCLCHECK(ncclAllReduce(buf.data() + off, buf.data() + off,
                                    count, ncclFloat, ncclSum, comm,
                                    get_comm_stream()));
            off += count;
        }
    };

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        if (rank == 0)
            std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        CUDACHECK(cudaMemsetAsync(accum_buf_upper.data(), 0,
                                  accum_buf_upper.bytes(), get_cuda_stream()));
        CUDACHECK(cudaMemsetAsync(accum_buf_lower.data(), 0,
                                  accum_buf_lower.bytes(), get_cuda_stream()));

        int batch_count = 0;
        int accum_count = 0;
        float epoch_compute_ms = 0.0f;
        float epoch_comm_ms = 0.0f;

        while (data_loader.next_batch(host_images, host_labels))
        {
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);

            CUDACHECK(cudaEventRecord(iter_start, get_cuda_stream()));
            float loss = model.forward(batch_x, batch_y);

            const bool is_final = (accum_count == accum_steps - 1);

            if (is_final)
            {
                // ---- Final step: overlap AllReduce with backward ----

                // Layer3 backward → pack → accumulate upper.
                model.backward_upper();
                pack(grad_buf_upper, upper_grads);
                accum_buf_upper.accumulate(grad_buf_upper, get_cuda_stream());
                CUDACHECK(cudaEventRecord(layer3_ready, get_cuda_stream()));

                // Comm stream: wait for upper grads, AllReduce in buckets.
                // Runs concurrently with layer2+1 backward on compute stream.
                CUDACHECK(cudaStreamWaitEvent(get_comm_stream(), layer3_ready, 0));
                CUDACHECK(cudaEventRecord(comm_start, get_comm_stream()));
                allreduce(accum_buf_upper, upper_elems);

                // Layer2+1 backward on compute stream (overlaps upper AllReduce).
                model.backward_lower();
                pack(grad_buf_lower, lower_grads);
                accum_buf_lower.accumulate(grad_buf_lower, get_cuda_stream());
                CUDACHECK(cudaEventRecord(compute_end, get_cuda_stream()));

                // Comm stream: wait for lower grads, AllReduce in buckets.
                CUDACHECK(cudaStreamWaitEvent(get_comm_stream(), compute_end, 0));
                allreduce(accum_buf_lower, lower_elems);
                CUDACHECK(cudaEventRecord(comm_end, get_comm_stream()));

                // Compute stream waits for all AllReduces before unpack/optimizer.
                CUDACHECK(cudaStreamWaitEvent(get_cuda_stream(), comm_end, 0));
                unpack(upper_grads, accum_buf_upper);
                unpack(lower_grads, accum_buf_lower);

                // Syncing compute stream is sufficient — comm_end has already
                // fired since compute stream waited for it.
                CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));

                float compute_ms, comm_ms;
                CUDACHECK(cudaEventElapsedTime(&compute_ms, iter_start, compute_end));
                CUDACHECK(cudaEventElapsedTime(&comm_ms, comm_start, comm_end));
                epoch_compute_ms += compute_ms;
                epoch_comm_ms += comm_ms;

                const float scale = static_cast<float>(world_size * accum_steps);
                for (FFLayer *layer : model.layers())
                {
                    layer->weight_grads().div(scale);
                    layer->bias_grads().div(scale);
                }

                optimizer.step(model.layer3());
                optimizer.step(model.layer2());
                optimizer.step(model.layer1());

                CUDACHECK(cudaMemsetAsync(accum_buf_upper.data(), 0,
                                          accum_buf_upper.bytes(), get_cuda_stream()));
                CUDACHECK(cudaMemsetAsync(accum_buf_lower.data(), 0,
                                          accum_buf_lower.bytes(), get_cuda_stream()));
                accum_count = 0;
            }
            else
            {
                // ---- Non-final step: accumulate without communicating ----
                model.backward();
                pack(grad_buf_upper, upper_grads);
                accum_buf_upper.accumulate(grad_buf_upper, get_cuda_stream());
                pack(grad_buf_lower, lower_grads);
                accum_buf_lower.accumulate(grad_buf_lower, get_cuda_stream());
                ++accum_count;
            }

            ++batch_count;

            if (rank == 0 && batch_count % 100 == 0)
            {
                std::printf("  Batch %d | loss = %.6f | acc = %.4f\n",
                            batch_count, loss, model.accuracy());
            }
        }

        if (rank == 0)
        {
            // compute_ms and comm_ms overlap, so their sum overstates wall time.
            std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                        epoch + 1, model.avg_loss(), model.accuracy());
            std::printf("  compute = %.1f ms | comm = %.1f ms (overlap) | comm %% = %.1f%%\n",
                        epoch_compute_ms, epoch_comm_ms,
                        100.0f * epoch_comm_ms / (epoch_compute_ms + epoch_comm_ms));
        }
    }

    CUDACHECK(cudaEventDestroy(iter_start));
    CUDACHECK(cudaEventDestroy(layer3_ready));
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
