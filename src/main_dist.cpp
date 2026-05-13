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
#include <limits>
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
static constexpr std::size_t default_overlap_bucket_elems =
    (25 * 1024 * 1024) / sizeof(float);

enum class TrainMode {
    Sync,
    Overlap
};

const char *mode_name(TrainMode mode)
{
    return mode == TrainMode::Overlap ? "overlap" : "sync";
}

struct GradSpec {
    Tensor *tensor;
    std::size_t flat_offset;
    int stage;
};

struct BucketSegment {
    Tensor *tensor;
    std::size_t tensor_offset;
    std::size_t flat_offset;
    std::size_t count;
    int stage;
};

struct GradBucket {
    std::size_t flat_offset = 0;
    std::size_t count = 0;
    int ready_stage = 0;
    cudaEvent_t comm_start = nullptr;
    cudaEvent_t comm_end = nullptr;
    std::vector<BucketSegment> segments;
};

std::vector<GradBucket> build_buckets(const std::vector<GradSpec>& specs,
                                      std::size_t bucket_size)
{
    std::vector<GradBucket> buckets;
    std::size_t flat_offset = 0;

    for (const GradSpec& spec : specs)
    {
        std::size_t tensor_offset = 0;
        std::size_t remaining = spec.tensor->size();

        while (remaining > 0)
        {
            if (buckets.empty() || buckets.back().count == bucket_size)
            {
                GradBucket bucket;
                bucket.flat_offset = flat_offset;
                bucket.ready_stage = spec.stage;
                buckets.push_back(std::move(bucket));
            }

            GradBucket& bucket = buckets.back();
            const std::size_t space = bucket_size - bucket.count;
            const std::size_t count = std::min(space, remaining);

            bucket.segments.push_back(BucketSegment{
                spec.tensor,
                tensor_offset,
                flat_offset,
                count,
                spec.stage
            });
            bucket.count += count;
            bucket.ready_stage = std::max(bucket.ready_stage, spec.stage);

            tensor_offset += count;
            flat_offset += count;
            remaining -= count;
        }
    }

    return buckets;
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
           int hidden_dim, long long bucket_size_arg, int accum_steps,
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

    std::vector<GradSpec> grad_specs = {
        {&model.layer3().weight_grads(), 0, 0},
        {&model.layer3().bias_grads(), 0, 0},
        {&model.layer2().weight_grads(), 0, 1},
        {&model.layer2().bias_grads(), 0, 1},
        {&model.layer1().weight_grads(), 0, 2},
        {&model.layer1().bias_grads(), 0, 2}
    };

    std::size_t total_elems = 0;
    for (GradSpec& spec : grad_specs)
    {
        spec.flat_offset = total_elems;
        total_elems += spec.tensor->size();
    }

    if (total_elems > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        if (rank == 0)
            std::printf("Total gradient buffer is too large for Tensor dimensions: %zu elements\n",
                        total_elems);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    std::size_t bucket_size = 0;
    if (bucket_size_arg <= 0)
        bucket_size = (mode == TrainMode::Overlap) ? default_overlap_bucket_elems : total_elems;
    else
        bucket_size = static_cast<std::size_t>(bucket_size_arg);

    bucket_size = std::max<std::size_t>(1, std::min(bucket_size, total_elems));

    std::vector<GradBucket> buckets = build_buckets(grad_specs, bucket_size);

    Tensor accum_buf(1, static_cast<int>(total_elems));

    if (rank == 0)
        std::printf("Total grad elems: %zu | bucket_size: %zu (%zu buckets)\n",
                    total_elems, bucket_size,
                    buckets.size());

    for (GradBucket& bucket : buckets)
    {
        CUDACHECK(cudaEventCreate(&bucket.comm_start));
        CUDACHECK(cudaEventCreate(&bucket.comm_end));
    }

    cudaEvent_t iter_start, iter_end, compute_end;
    cudaEvent_t wait_start, wait_end, comm_done;
    cudaEvent_t stage_ready[3];
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&iter_end));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&wait_start));
    CUDACHECK(cudaEventCreate(&wait_end));
    CUDACHECK(cudaEventCreateWithFlags(&comm_done, cudaEventDisableTiming));
    for (cudaEvent_t& event : stage_ready)
        CUDACHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    auto accumulate_stage = [&](int stage)
    {
        for (const GradBucket& bucket : buckets)
        {
            for (const BucketSegment& segment : bucket.segments)
            {
                if (segment.stage == stage)
                {
                    accum_buf.accumulate_slice(*segment.tensor,
                                               segment.flat_offset,
                                               segment.tensor_offset,
                                               segment.count,
                                               compute_stream);
                }
            }
        }
    };

    auto unpack = [&]()
    {
        for (const GradSpec& spec : grad_specs)
        {
            CUDACHECK(cudaMemcpyAsync(spec.tensor->data(),
                                      accum_buf.data() + spec.flat_offset,
                                      spec.tensor->size() * sizeof(float),
                                      cudaMemcpyDeviceToDevice,
                                      compute_stream));
        }
    };

    auto launch_sync_allreduce = [&]()
    {
        for (GradBucket& bucket : buckets)
        {
            CUDACHECK(cudaEventRecord(bucket.comm_start, compute_stream));
            NCCLCHECK(ncclAllReduce(accum_buf.data() + bucket.flat_offset,
                                    accum_buf.data() + bucket.flat_offset,
                                    bucket.count, ncclFloat, ncclSum, comm,
                                    compute_stream));
            CUDACHECK(cudaEventRecord(bucket.comm_end, compute_stream));
        }
    };

    auto launch_overlap_buckets = [&](int stage, bool final_accum)
    {
        if (!final_accum)
            return;

        for (GradBucket& bucket : buckets)
        {
            if (bucket.ready_stage != stage)
                continue;

            CUDACHECK(cudaStreamWaitEvent(comm_stream, stage_ready[stage], 0));
            nvtxRangePushA("overlap/comm_bucket");
            CUDACHECK(cudaEventRecord(bucket.comm_start, comm_stream));
            NCCLCHECK(ncclAllReduce(accum_buf.data() + bucket.flat_offset,
                                    accum_buf.data() + bucket.flat_offset,
                                    bucket.count, ncclFloat, ncclSum, comm,
                                    comm_stream));
            CUDACHECK(cudaEventRecord(bucket.comm_end, comm_stream));
            nvtxRangePop();
        }
    };

    auto scale_and_step = [&]()
    {
        const float scale = static_cast<float>(world_size * accum_steps);
        for (FFLayer *layer : model.layers())
        {
            layer->weight_grads().div(scale, compute_stream);
            layer->bias_grads().div(scale, compute_stream);
        }

        optimizer.step(model.layer3(), compute_stream);
        optimizer.step(model.layer2(), compute_stream);
        optimizer.step(model.layer1(), compute_stream);
    };

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        if (rank == 0)
            std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        CUDACHECK(cudaMemsetAsync(accum_buf.data(), 0,
                                  accum_buf.bytes(), compute_stream));

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
                accumulate_stage(0);
                accumulate_stage(1);
                accumulate_stage(2);
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
                accumulate_stage(0);
                CUDACHECK(cudaEventRecord(stage_ready[0], compute_stream));
                nvtxRangePop();
                launch_overlap_buckets(0, final_accum);

                nvtxRangePushA("overlap/bwd_layer3_input");
                model.backward_layer3_input();
                nvtxRangePop();

                nvtxRangePushA("overlap/bwd_layer2_grads");
                model.backward_layer2_grads();
                accumulate_stage(1);
                CUDACHECK(cudaEventRecord(stage_ready[1], compute_stream));
                nvtxRangePop();
                launch_overlap_buckets(1, final_accum);

                nvtxRangePushA("overlap/bwd_layer2_input");
                model.backward_layer2_input();
                nvtxRangePop();

                nvtxRangePushA("overlap/bwd_layer1_grads");
                model.backward_layer1_grads();
                accumulate_stage(2);
                CUDACHECK(cudaEventRecord(stage_ready[2], compute_stream));
                nvtxRangePop();
                launch_overlap_buckets(2, final_accum);

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
                nvtxRangePushA(mode == TrainMode::Overlap ? "overlap/unpack" : "sync/unpack");
                unpack();
                nvtxRangePop();

                nvtxRangePushA(mode == TrainMode::Overlap ? "overlap/opt" : "sync/opt");
                scale_and_step();
                nvtxRangePop();

                CUDACHECK(cudaMemsetAsync(accum_buf.data(), 0,
                                          accum_buf.bytes(), compute_stream));
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
                for (const GradBucket& bucket : buckets)
                {
                    float bucket_ms = 0.0f;
                    CUDACHECK(cudaEventElapsedTime(&bucket_ms,
                                                  bucket.comm_start,
                                                  bucket.comm_end));
                    comm_total_ms += bucket_ms;
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
    for (cudaEvent_t& event : stage_ready)
        CUDACHECK(cudaEventDestroy(event));
    for (GradBucket& bucket : buckets)
    {
        CUDACHECK(cudaEventDestroy(bucket.comm_start));
        CUDACHECK(cudaEventDestroy(bucket.comm_end));
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

    // argv[1]: dataset     ("mnist" default | "cifar10")
    // argv[2]: hidden_dim  (default 1024 for mnist, 4096 for cifar10)
    // argv[3]: bucket_size (default -1: sync=one bucket, overlap=25 MiB buckets)
    // argv[4]: accum_steps (default 1 = no accumulation)
    // argv[5]: algo        ("ring", "tree", or omit to let NCCL decide)
    // argv[6]: learning_rate (default 0.3 for mnist, 0.01 for cifar10)
    // argv[7]: mode        ("overlap" default | "sync")
    // argv[8]: epochs      (default 10)
    const char *dataset_arg = (argc >= 2) ? argv[1] : "mnist";
    bool use_cifar10 = (std::strcmp(dataset_arg, "cifar10") == 0);
    int default_hidden = use_cifar10 ? 4096 : 1024;
    int hidden_dim  = (argc >= 3) ? std::atoi(argv[2]) : default_hidden;
    long long bucket_size = (argc >= 4) ? std::atoll(argv[3]) : -1;
    int accum_steps = (argc >= 5) ? std::atoi(argv[4]) : 1;
    const char *algo_arg = (argc >= 6) ? argv[5] : "auto";
    float default_lr = use_cifar10 ? 0.01f : 0.3f;
    float learning_rate = (argc >= 7) ? std::atof(argv[6]) : default_lr;
    const char *mode_arg = (argc >= 8) ? argv[7] : "overlap";
    int epochs = (argc >= 9) ? std::atoi(argv[8]) : default_num_epochs;

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
              hidden_dim, bucket_size, accum_steps, algo_arg, learning_rate,
              mode, epochs);
    } else {
        MNIST train_data(std::string(data_dir) + "/train-images-idx3-ubyte",
                         std::string(data_dir) + "/train-labels-idx1-ubyte");
        train(rank, size, train_data, output_path, comm,
              MNIST::IMAGE_DIM, MNIST::NUM_CLASSES,
              hidden_dim, bucket_size, accum_steps, algo_arg, learning_rate,
              mode, epochs);
    }

    ncclCommDestroy(comm);
    MPICHECK(MPI_Finalize());
    return 0;
}
