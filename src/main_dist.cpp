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

static constexpr int batch_size = 80;
static constexpr int input_dim = 784;
static constexpr int output_dim = 10;
static constexpr int num_epochs = 5;

Model create_model(int rank, int local_batch_size, int hidden_dim, unsigned int seed = 0)
{
    Model model(input_dim, hidden_dim, output_dim, local_batch_size);
    // only initialize on rank 0 and broadcast model params to others
    if (rank == 0)
    {
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
           const char *output_path, ncclComm_t comm, int hidden_dim,
           int bucket_size, int accum_steps, const char *algo)
{
    std::printf("Executing training routine\n");
    if (rank == 0)
        std::printf("NCCL algorithm: %s | accum_steps: %d\n", algo, accum_steps);

    std::string base(data_dir);

    MNIST train_data(base + "/train-images-idx3-ubyte",
                     base + "/train-labels-idx1-ubyte");

    SGDOptimizer optimizer(0.3f);

    const int local_batch_size = batch_size / world_size;
    Model model = create_model(rank, local_batch_size, hidden_dim, 42);

    std::vector<float> host_images, host_labels;

    Tensor batch_x(local_batch_size, input_dim);
    Tensor batch_y(local_batch_size, output_dim);

    DistributedDataLoader data_loader(train_data, rank, world_size, batch_size, 42);

    // Separate grad groups so we can AllReduce layer2 (upper) while layer1
    // (lower) backward is still running on the compute stream.
    std::vector<Tensor *> upper_grads = {&model.layer2().weight_grads(),
                                         &model.layer2().bias_grads()};
    std::vector<Tensor *> lower_grads = {&model.layer1().weight_grads(),
                                         &model.layer1().bias_grads()};

    int upper_elems = 0, lower_elems = 0;
    for (Tensor *t : upper_grads) upper_elems += static_cast<int>(t->size());
    for (Tensor *t : lower_grads) lower_elems += static_cast<int>(t->size());

    if (bucket_size <= 0)
        bucket_size = upper_elems + lower_elems;

    // One scratch buffer and one accumulation buffer per layer group.
    Tensor grad_buf_upper(1, upper_elems), accum_buf_upper(1, upper_elems);
    Tensor grad_buf_lower(1, lower_elems), accum_buf_lower(1, lower_elems);

    if (rank == 0)
        std::printf("Grad elems: upper(layer2)=%d lower(layer1)=%d bucket_size=%d\n",
                    upper_elems, lower_elems, bucket_size);

    (void)get_comm_stream();  // warm up before training loop

    // iter_start/compute_end on compute stream for compute timing.
    // layer2_ready: signals comm stream that upper grads are in accum_buf_upper.
    // comm_start/comm_end on comm stream for AllReduce timing.
    cudaEvent_t iter_start, layer2_ready, compute_end, comm_start, comm_end;
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&layer2_ready));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&comm_start));
    CUDACHECK(cudaEventCreate(&comm_end));

    // Lambda: pack a list of grad tensors into a flat buffer starting at offset 0.
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

    // Lambda: unpack a flat buffer back into a list of grad tensors.
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

    // Lambda: AllReduce a flat buffer in bucket_size chunks on the comm stream.
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

                // Layer2 backward → pack → accumulate into upper buffer.
                model.backward_upper();
                pack(grad_buf_upper, upper_grads);
                accum_buf_upper.accumulate(grad_buf_upper, get_cuda_stream());
                CUDACHECK(cudaEventRecord(layer2_ready, get_cuda_stream()));

                // Comm stream: wait for upper grads, AllReduce them.
                // Runs concurrently with layer1 backward on compute stream.
                CUDACHECK(cudaStreamWaitEvent(get_comm_stream(), layer2_ready, 0));
                CUDACHECK(cudaEventRecord(comm_start, get_comm_stream()));
                allreduce(accum_buf_upper, upper_elems);

                // Layer1 backward on compute stream (overlaps upper AllReduce).
                model.backward_lower();
                pack(grad_buf_lower, lower_grads);
                accum_buf_lower.accumulate(grad_buf_lower, get_cuda_stream());
                CUDACHECK(cudaEventRecord(compute_end, get_cuda_stream()));

                // Comm stream: wait for lower grads, AllReduce them.
                CUDACHECK(cudaStreamWaitEvent(get_comm_stream(), compute_end, 0));
                allreduce(accum_buf_lower, lower_elems);
                CUDACHECK(cudaEventRecord(comm_end, get_comm_stream()));

                // Compute stream waits for all AllReduces before unpack/optimizer.
                CUDACHECK(cudaStreamWaitEvent(get_cuda_stream(), comm_end, 0));
                unpack(upper_grads, accum_buf_upper);
                unpack(lower_grads, accum_buf_lower);

                // Sync compute stream — comm_end has already fired since compute
                // stream waited for it, so one sync covers both streams.
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
            // compute_ms and comm_ms overlap on the final step of each window,
            // so their sum overstates wall time by the amount of overlap gained.
            std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                        epoch + 1, model.avg_loss(), model.accuracy());
            std::printf("  compute = %.1f ms | comm = %.1f ms (overlap) | comm %% = %.1f%%\n",
                        epoch_compute_ms, epoch_comm_ms,
                        100.0f * epoch_comm_ms / (epoch_compute_ms + epoch_comm_ms));
        }
    }

    CUDACHECK(cudaEventDestroy(iter_start));
    CUDACHECK(cudaEventDestroy(layer2_ready));
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

    // argv[1]: hidden_dim  (default 32)
    // argv[2]: bucket_size (default -1 = one bucket)
    // argv[3]: accum_steps (default 1 = no accumulation)
    // argv[4]: algo        ("ring", "tree", or omit to let NCCL decide)
    int hidden_dim = (argc >= 2) ? std::atoi(argv[1]) : 32;
    int bucket_size = (argc >= 3) ? std::atoi(argv[2]) : -1;
    int accum_steps = (argc >= 4) ? std::atoi(argv[3]) : 1;
    const char *algo_arg = (argc >= 5) ? argv[4] : "auto";

    if (accum_steps < 1)
        accum_steps = 1;

    // Must be set before ncclCommInitRank so NCCL picks up the algorithm.
    if (std::strcmp(algo_arg, "ring") == 0)
        setenv("NCCL_ALGO", "Ring", 1);
    else if (std::strcmp(algo_arg, "tree") == 0)
        setenv("NCCL_ALGO", "Tree", 1);
    // "auto" or anything else: leave NCCL_ALGO unset so NCCL decides.

    // Slurm should already handle GPU assignment where it assigns one GPU per
    // rank
    int local_rank = 0;
    const char *local_rank_env = std::getenv("SLURM_LOCALID");
    if (local_rank_env != nullptr)
    {
        local_rank = std::atoi(local_rank_env);
    }
    CUDACHECK(cudaSetDevice(local_rank));
    // Warm up the lazy stream singleton so NCCL and kernels share it.
    (void)get_cuda_stream();

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    const char data_dir[] = "data";
    const char output_path[] = "ff.params";

    train(rank, size, data_dir, output_path, comm, hidden_dim, bucket_size, accum_steps, algo_arg);

    ncclCommDestroy(comm);
    // Stream is owned by the get_cuda_stream() static and is released at exit.
    MPICHECK(MPI_Finalize());
    return 0;
}