#include "CIFAR10.hpp"
#include "DistributedDataLoader.hpp"
#include "FFLayer.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include "SGDOptimizer.hpp"
#include "Stream.hpp"
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>
#include <utility>
#include <vector>

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
static constexpr int default_num_epochs = 10;

enum class TrainMode {
    Sync,
    Overlap
};

const char *mode_name(TrainMode mode)
{
    return mode == TrainMode::Overlap ? "overlap" : "sync";
}

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
           int hidden_dim, int accum_steps,
           const char *algo, float learning_rate, TrainMode mode, int epochs)
{
    std::printf("Executing training routine\n");
    if (rank == 0)
        std::printf("mode: %s | NCCL algorithm: %s | epochs: %d | accum_steps: %d | input_dim: %d | hidden_dim: %d | lr: %.4f\n",
                    mode_name(mode), algo, epochs, accum_steps, input_dim, hidden_dim, learning_rate);

    cudaStream_t compute_stream = get_cuda_stream();
    cudaStream_t comm_stream = get_comm_stream();
    nvtxNameCudaStreamA(compute_stream, "compute");
    nvtxNameCudaStreamA(comm_stream, "comm");

    SGDOptimizer optimizer(learning_rate);

    const int local_batch_size = batch_size / world_size;
    Model model = create_model(rank, local_batch_size, input_dim, output_dim, hidden_dim, 42);

    std::vector<float> host_images, host_labels;

    Tensor batch_x(local_batch_size, input_dim);
    Tensor batch_y(local_batch_size, output_dim);

    DistributedDataLoader data_loader(train_data, rank, world_size, batch_size, 42);

    cudaEvent_t iter_start, iter_end, compute_end;
    cudaEvent_t wait_start, wait_end, comm_done;
    cudaEvent_t layer_ready[3];
    cudaEvent_t layer_comm_start[3], layer_comm_end[3];
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&iter_end));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&wait_start));
    CUDACHECK(cudaEventCreate(&wait_end));
    CUDACHECK(cudaEventCreateWithFlags(&comm_done, cudaEventDisableTiming));
    for (cudaEvent_t& event : layer_ready)
        CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    for (int i = 0; i < 3; ++i)
    {
        CUDACHECK(cudaEventCreate(&layer_comm_start[i]));
        CUDACHECK(cudaEventCreate(&layer_comm_end[i]));
    }

    // Layers are allreduced in backward order: layer3 first (slot 0),
    // then layer2 (slot 1), then layer1 (slot 2).
    FFLayer* ordered_layers[3] = {&model.layer3(), &model.layer2(), &model.layer1()};

    auto launch_sync_allreduce = [&]()
    {
        for (int i = 0; i < 3; ++i)
        {
            FFLayer* layer = ordered_layers[i];
            CUDACHECK(cudaEventRecord(layer_comm_start[i], compute_stream));
            NCCLCHECK(ncclAllReduce(layer->weight_grad_accum().data(),
                                    layer->weight_grad_accum().data(),
                                    layer->weight_grad_accum().size(),
                                    ncclFloat, ncclSum, comm, compute_stream));
            NCCLCHECK(ncclAllReduce(layer->bias_grad_accum().data(),
                                    layer->bias_grad_accum().data(),
                                    layer->bias_grad_accum().size(),
                                    ncclFloat, ncclSum, comm, compute_stream));
            CUDACHECK(cudaEventRecord(layer_comm_end[i], compute_stream));
        }
    };

    auto launch_overlap_layer = [&](int slot, bool final_accum)
    {
        if (!final_accum)
            return;

        FFLayer* layer = ordered_layers[slot];
        CUDACHECK(cudaStreamWaitEvent(comm_stream, layer_ready[slot], 0));
        nvtxRangePushA("overlap/comm_layer");
        CUDACHECK(cudaEventRecord(layer_comm_start[slot], comm_stream));
        NCCLCHECK(ncclAllReduce(layer->weight_grad_accum().data(),
                                layer->weight_grad_accum().data(),
                                layer->weight_grad_accum().size(),
                                ncclFloat, ncclSum, comm, comm_stream));
        NCCLCHECK(ncclAllReduce(layer->bias_grad_accum().data(),
                                layer->bias_grad_accum().data(),
                                layer->bias_grad_accum().size(),
                                ncclFloat, ncclSum, comm, comm_stream));
        CUDACHECK(cudaEventRecord(layer_comm_end[slot], comm_stream));
        nvtxRangePop();
    };

    auto scale_and_step = [&]()
    {
        const float scale = static_cast<float>(world_size * accum_steps);
        for (FFLayer *layer : model.layers())
        {
            layer->weight_grad_accum().div(scale, compute_stream);
            layer->bias_grad_accum().div(scale, compute_stream);
        }

        optimizer.step(model.layer3(),
                       model.layer3().weight_grad_accum(),
                       model.layer3().bias_grad_accum(), compute_stream);
        optimizer.step(model.layer2(),
                       model.layer2().weight_grad_accum(),
                       model.layer2().bias_grad_accum(), compute_stream);
        optimizer.step(model.layer1(),
                       model.layer1().weight_grad_accum(),
                       model.layer1().bias_grad_accum(), compute_stream);
    };

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (rank == 0)
            std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        for (FFLayer *layer : model.layers())
            layer->zero_grad_accum(compute_stream);

        int batch_count = 0;
        int accum_count = 0;
        float epoch_iter_ms = 0.0f;
        float epoch_compute_ms = 0.0f;
        float epoch_comm_total_ms = 0.0f;
        float epoch_comm_exposed_ms = 0.0f;

        while (data_loader.next_batch(host_images, host_labels))
        {
            nvtxRangePushA(mode == TrainMode::Overlap ? "overlap/iter" : "sync/iter");

            nvtxRangePushA("h2d");
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);
            nvtxRangePop();

            CUDACHECK(cudaEventRecord(iter_start, compute_stream));

            nvtxRangePushA(mode == TrainMode::Overlap ? "overlap/fwd" : "sync/fwd");
            float loss = model.forward(batch_x, batch_y);
            nvtxRangePop();

            ++accum_count;
            const bool final_accum = (accum_count == accum_steps);

            if (mode == TrainMode::Sync)
            {
                nvtxRangePushA("sync/bwd");
                model.backward();
                nvtxRangePop();

                nvtxRangePushA("sync/accumulate");
                for (FFLayer *layer : model.layers())
                    layer->accumulate_grads(compute_stream);
                nvtxRangePop();

                CUDACHECK(cudaEventRecord(compute_end, compute_stream));

                if (final_accum)
                {
                    nvtxRangePushA("sync/allreduce");
                    launch_sync_allreduce();
                    nvtxRangePop();
                }
            }
            else
            {
                nvtxRangePushA("overlap/bwd_loss");
                model.backward_loss();
                nvtxRangePop();

                nvtxRangePushA("overlap/bwd_layer3_grads");
                model.backward_layer3_grads();
                model.layer3().accumulate_grads(compute_stream);
                CUDACHECK(cudaEventRecord(layer_ready[0], compute_stream));
                nvtxRangePop();
                launch_overlap_layer(0, final_accum);

                nvtxRangePushA("overlap/bwd_layer3_input");
                model.backward_layer3_input();
                nvtxRangePop();

                nvtxRangePushA("overlap/bwd_layer2_grads");
                model.backward_layer2_grads();
                model.layer2().accumulate_grads(compute_stream);
                CUDACHECK(cudaEventRecord(layer_ready[1], compute_stream));
                nvtxRangePop();
                launch_overlap_layer(1, final_accum);

                nvtxRangePushA("overlap/bwd_layer2_input");
                model.backward_layer2_input();
                nvtxRangePop();

                nvtxRangePushA("overlap/bwd_layer1_grads");
                model.backward_layer1_grads();
                model.layer1().accumulate_grads(compute_stream);
                CUDACHECK(cudaEventRecord(layer_ready[2], compute_stream));
                nvtxRangePop();
                launch_overlap_layer(2, final_accum);

                nvtxRangePushA("overlap/bwd_layer1_input");
                model.backward_layer1_input();
                nvtxRangePop();

                CUDACHECK(cudaEventRecord(compute_end, compute_stream));

                if (final_accum)
                {
                    CUDACHECK(cudaEventRecord(comm_done, comm_stream));

                    nvtxRangePushA("overlap/wait_comm");
                    CUDACHECK(cudaEventRecord(wait_start, compute_stream));
                    CUDACHECK(cudaStreamWaitEvent(compute_stream, comm_done, 0));
                    CUDACHECK(cudaEventRecord(wait_end, compute_stream));
                    nvtxRangePop();
                }
            }

            if (final_accum)
            {
                nvtxRangePushA(mode == TrainMode::Overlap ? "overlap/opt" : "sync/opt");
                scale_and_step();
                nvtxRangePop();

                // sgd_update_kernel zeroes the grad accumulators as a side effect,
                // so they are ready for the next accumulation period.
                accum_count = 0;
            }

            ++batch_count;

            CUDACHECK(cudaEventRecord(iter_end, compute_stream));
            CUDACHECK(cudaEventSynchronize(iter_end));

            float iter_ms = 0.0f;
            float compute_ms = 0.0f;
            float comm_total_ms = 0.0f;
            float comm_exposed_ms = 0.0f;
            CUDACHECK(cudaEventElapsedTime(&iter_ms, iter_start, iter_end));
            CUDACHECK(cudaEventElapsedTime(&compute_ms, iter_start, compute_end));

            if (final_accum)
            {
                // In overlap mode the per-layer comm events live on comm_stream.
                // Wait on comm_done explicitly so the elapsed-time reads below
                // can never see a not-yet-completed event ("device not ready").
                if (mode == TrainMode::Overlap)
                    CUDACHECK(cudaEventSynchronize(comm_done));

                for (int i = 0; i < 3; ++i)
                {
                    float layer_ms = 0.0f;
                    CUDACHECK(cudaEventElapsedTime(&layer_ms,
                                                  layer_comm_start[i],
                                                  layer_comm_end[i]));
                    comm_total_ms += layer_ms;
                }

                if (mode == TrainMode::Overlap)
                    CUDACHECK(cudaEventElapsedTime(&comm_exposed_ms, wait_start, wait_end));
                else
                    comm_exposed_ms = comm_total_ms;
            }

            epoch_iter_ms += iter_ms;
            epoch_compute_ms += compute_ms;
            epoch_comm_total_ms += comm_total_ms;
            epoch_comm_exposed_ms += comm_exposed_ms;

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
            const float hidden_comm_ms =
                std::max(0.0f, epoch_comm_total_ms - epoch_comm_exposed_ms);
            const float hidden_pct =
                epoch_comm_total_ms > 0.0f
                    ? 100.0f * hidden_comm_ms / epoch_comm_total_ms
                    : 0.0f;
            std::printf("  iter = %.1f ms | compute = %.1f ms | comm_total = %.1f ms | comm_exposed_wait = %.1f ms | hidden_comm = %.1f%%\n",
                        epoch_iter_ms, epoch_compute_ms, epoch_comm_total_ms,
                        epoch_comm_exposed_ms, hidden_pct);
        }
    }

    CUDACHECK(cudaEventDestroy(iter_start));
    CUDACHECK(cudaEventDestroy(iter_end));
    CUDACHECK(cudaEventDestroy(compute_end));
    CUDACHECK(cudaEventDestroy(wait_start));
    CUDACHECK(cudaEventDestroy(wait_end));
    CUDACHECK(cudaEventDestroy(comm_done));
    for (cudaEvent_t& event : layer_ready)
        CUDACHECK(cudaEventDestroy(event));
    for (int i = 0; i < 3; ++i)
    {
        CUDACHECK(cudaEventDestroy(layer_comm_start[i]));
        CUDACHECK(cudaEventDestroy(layer_comm_end[i]));
    }

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

    // argv[1]: dataset       ("mnist" default | "cifar10")
    // argv[2]: hidden_dim    (default 1024 for mnist, 4096 for cifar10)
    // argv[3]: accum_steps   (default 1 = no accumulation)
    // argv[4]: algo          ("ring", "tree", or omit to let NCCL decide)
    // argv[5]: learning_rate (default 0.3 for mnist, 0.01 for cifar10)
    // argv[6]: mode          ("overlap" default | "sync")
    // argv[7]: epochs        (default 10)
    const char *dataset_arg = (argc >= 2) ? argv[1] : "mnist";
    bool use_cifar10 = (std::strcmp(dataset_arg, "cifar10") == 0);
    int default_hidden = use_cifar10 ? 4096 : 1024;
    int hidden_dim  = (argc >= 3) ? std::atoi(argv[2]) : default_hidden;
    int accum_steps = (argc >= 4) ? std::atoi(argv[3]) : 1;
    const char *algo_arg = (argc >= 5) ? argv[4] : "auto";
    float default_lr = use_cifar10 ? 0.01f : 0.3f;
    float learning_rate = (argc >= 6) ? std::atof(argv[5]) : default_lr;
    const char *mode_arg = (argc >= 7) ? argv[6] : "overlap";
    int epochs = (argc >= 8) ? std::atoi(argv[7]) : default_num_epochs;

    TrainMode mode = TrainMode::Overlap;
    if (std::strcmp(mode_arg, "sync") == 0)
        mode = TrainMode::Sync;
    else if (std::strcmp(mode_arg, "overlap") != 0)
    {
        if (rank == 0)
            std::printf("Unknown mode '%s'. Expected 'sync' or 'overlap'.\n", mode_arg);
        MPICHECK(MPI_Finalize());
        return EXIT_FAILURE;
    }

    if (accum_steps < 1)
        accum_steps = 1;
    if (epochs < 1)
        epochs = 1;

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
    (void)get_comm_stream();

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    const char data_dir[] = "data";
    const char output_path[] = "ff.params";

    if (use_cifar10) {
        CIFAR10 train_data(std::string(data_dir) + "/cifar-10-batches-bin", true);
        train(rank, size, train_data, output_path, comm,
              CIFAR10::IMAGE_DIM, CIFAR10::NUM_CLASSES,
              hidden_dim, accum_steps, algo_arg, learning_rate,
              mode, epochs);
    } else {
        MNIST train_data(std::string(data_dir) + "/train-images-idx3-ubyte",
                         std::string(data_dir) + "/train-labels-idx1-ubyte");
        train(rank, size, train_data, output_path, comm,
              MNIST::IMAGE_DIM, MNIST::NUM_CLASSES,
              hidden_dim, accum_steps, algo_arg, learning_rate,
              mode, epochs);
    }

    ncclCommDestroy(comm);
    MPICHECK(MPI_Finalize());
    return 0;
}
