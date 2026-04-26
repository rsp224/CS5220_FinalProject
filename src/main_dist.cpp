#include "DistributedDataLoader.hpp"
#include "FFLayer.hpp"
#include "MNIST.hpp"
#include "Model.hpp"
#include "SGDOptimizer.hpp"
#include "Stream.hpp"
#include <algorithm>
#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>
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

static constexpr int batch_size = 80;
static constexpr int input_dim = 784;
static constexpr int hidden_dim = 32;
static constexpr int output_dim = 10;
static constexpr int num_epochs = 5;

Model create_model(int rank, int local_batch_size, unsigned int seed = 0)
{
    Model model(input_dim, hidden_dim, output_dim, local_batch_size);
    // only initialize on rank 0 and broadcast model params to others
    if (rank == 0)
    {
        model.init(seed);
    }

    auto broadcast_tensor = [rank](Tensor &tensor) {
        std::vector<float> host;
        if (rank == 0)
        {
            host = tensor.copy_to_host();
        }
        else
        {
            host.resize(tensor.size());
        }

        MPICHECK(MPI_Bcast(host.data(), static_cast<int>(host.size()),
                           MPI_FLOAT, 0, MPI_COMM_WORLD));
        tensor.copy_from_host(host);
    };

    broadcast_tensor(model.layer1().weights());
    broadcast_tensor(model.layer1().biases());
    broadcast_tensor(model.layer2().weights());
    broadcast_tensor(model.layer2().biases());

    return model;
}

void train(int rank, int world_size, const char *data_dir,
           const char *output_path, ncclComm_t comm, int bucket_size)
{
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

    struct GradEntry
    {
        Tensor *tensor;
        FFLayer *layer;
        int offset;
        int bucket_idx;
    };

    struct GradBucket
    {
        int offset;
        int count;
        int ready;
        bool launched;
        cudaEvent_t ready_event;
    };

    std::vector<FFLayer *> layers = model.layers();
    std::vector<GradEntry> grad_entries;
    for (auto layer_it = layers.rbegin(); layer_it != layers.rend(); ++layer_it)
    {
        FFLayer *layer = *layer_it;
        grad_entries.push_back({&layer->bias_grads(), layer, 0, -1});
        grad_entries.push_back({&layer->weight_grads(), layer, 0, -1});
    }

    int total_grad_elems = 0;
    for (const GradEntry &entry : grad_entries)
        total_grad_elems += static_cast<int>(entry.tensor->size());

    if (bucket_size <= 0)
        bucket_size = total_grad_elems;

    std::vector<GradBucket> buckets;
    int offset = 0;
    FFLayer *current_bucket_layer = nullptr;
    for (GradEntry &entry : grad_entries)
    {
        int count = static_cast<int>(entry.tensor->size());
        if (buckets.empty() ||
            (buckets.back().count > 0 &&
             (buckets.back().count + count > bucket_size ||
              current_bucket_layer != entry.layer)))
        {
            buckets.push_back({offset, 0, 0, false, nullptr});
            current_bucket_layer = entry.layer;
        }

        entry.offset = offset;
        entry.bucket_idx = static_cast<int>(buckets.size()) - 1;
        buckets.back().count += count;
        offset += count;
    }

    Tensor grad_buffer(1, total_grad_elems);

    cudaStream_t comm_stream;
    CUDACHECK(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));

    for (GradBucket &bucket : buckets)
    {
        CUDACHECK(cudaEventCreateWithFlags(&bucket.ready_event,
                                           cudaEventDisableTiming));
    }

    if (rank == 0)
        std::printf("Gradient bucketing: total=%d elems, bucket_size=%d (%zu buckets)\n",
                    total_grad_elems, bucket_size, buckets.size());

    cudaEvent_t iter_start, compute_end, comm_span_start, comm_span_end, sync_end;
    CUDACHECK(cudaEventCreate(&iter_start));
    CUDACHECK(cudaEventCreate(&compute_end));
    CUDACHECK(cudaEventCreate(&comm_span_start));
    CUDACHECK(cudaEventCreate(&comm_span_end));
    CUDACHECK(cudaEventCreate(&sync_end));

    for (int epoch = 0; epoch < num_epochs; ++epoch)
    {
        if (rank == 0)
            std::printf("Epoch %d\n", epoch + 1);

        data_loader.reset();
        model.reset_score();

        int batch_count = 0;
        float epoch_compute_ms = 0.0f;
        float epoch_comm_ms = 0.0f;
        float epoch_total_ms = 0.0f;
        float epoch_overlap_ms = 0.0f;

        while (data_loader.next_batch(host_images, host_labels))
        {
            batch_x.copy_from_host(host_images);
            batch_y.copy_from_host(host_labels);

            for (GradBucket &bucket : buckets)
            {
                bucket.ready = 0;
                bucket.launched = false;
            }

            bool comm_started = false;

            CUDACHECK(cudaEventRecord(iter_start, get_cuda_stream()));

            float loss = model.forward(batch_x, batch_y);
            model.backward([&](FFLayer &completed_layer) {
                for (GradEntry &entry : grad_entries)
                {
                    if (entry.layer != &completed_layer)
                        continue;

                    GradBucket &bucket = buckets[entry.bucket_idx];
                    bucket.ready += static_cast<int>(entry.tensor->size());

                    if (bucket.ready == bucket.count && !bucket.launched)
                    {
                        CUDACHECK(cudaEventRecord(bucket.ready_event,
                                                  get_cuda_stream()));
                        CUDACHECK(cudaStreamWaitEvent(comm_stream,
                                                      bucket.ready_event, 0));

                        if (!comm_started)
                        {
                            CUDACHECK(cudaEventRecord(comm_span_start,
                                                      comm_stream));
                            comm_started = true;
                        }

                        for (const GradEntry &bucket_entry : grad_entries)
                        {
                            if (bucket_entry.bucket_idx != entry.bucket_idx)
                                continue;

                            CUDACHECK(cudaMemcpyAsync(
                                grad_buffer.data() + bucket_entry.offset,
                                bucket_entry.tensor->data(),
                                bucket_entry.tensor->size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, comm_stream));
                        }

                        NCCLCHECK(ncclAllReduce(grad_buffer.data() + bucket.offset,
                                                grad_buffer.data() + bucket.offset,
                                                bucket.count, ncclFloat, ncclSum,
                                                comm, comm_stream));

                        for (const GradEntry &bucket_entry : grad_entries)
                        {
                            if (bucket_entry.bucket_idx != entry.bucket_idx)
                                continue;

                            CUDACHECK(cudaMemcpyAsync(
                                bucket_entry.tensor->data(),
                                grad_buffer.data() + bucket_entry.offset,
                                bucket_entry.tensor->size() * sizeof(float),
                                cudaMemcpyDeviceToDevice, comm_stream));
                        }

                        bucket.launched = true;
                    }
                }
            });

            CUDACHECK(cudaEventRecord(compute_end, get_cuda_stream()));

            for (const GradBucket &bucket : buckets)
            {
                if (!bucket.launched)
                {
                    std::printf("Rank %d failed to launch gradient bucket at offset %d\n",
                                rank, bucket.offset);
                    exit(EXIT_FAILURE);
                }
            }

            if (comm_started)
            {
                CUDACHECK(cudaEventRecord(comm_span_end, comm_stream));
                CUDACHECK(cudaStreamWaitEvent(get_cuda_stream(), comm_span_end,
                                              0));
            }

            CUDACHECK(cudaEventRecord(sync_end, get_cuda_stream()));

            // Sync stream — all events and memcpys complete after this
            CUDACHECK(cudaStreamSynchronize(get_cuda_stream()));

            float compute_ms = 0.0f;
            float comm_ms = 0.0f;
            float total_ms = 0.0f;
            CUDACHECK(cudaEventElapsedTime(&compute_ms, iter_start, compute_end));
            if (comm_started)
                CUDACHECK(cudaEventElapsedTime(&comm_ms, comm_span_start,
                                               comm_span_end));
            CUDACHECK(cudaEventElapsedTime(&total_ms, iter_start, sync_end));
            epoch_compute_ms += compute_ms;
            epoch_comm_ms += comm_ms;
            epoch_total_ms += total_ms;
            epoch_overlap_ms += std::max(0.0f, compute_ms + comm_ms - total_ms);

            for (FFLayer *layer : model.layers())
            {
                layer->weight_grads().div(world_size);
                layer->bias_grads().div(world_size);
            }

            optimizer.step(model.layer2());
            optimizer.step(model.layer1());

            ++batch_count;

            if (rank == 0 && batch_count % 100 == 0)
            {
                std::printf("  Batch %d | loss = %.6f | acc = %.4f\n",
                            batch_count, loss, model.accuracy());
            }
        }

        if (rank == 0)
        {
            float comm_pct =
                epoch_total_ms > 0.0f ? 100.0f * epoch_comm_ms / epoch_total_ms
                                      : 0.0f;
            std::printf("Epoch %d complete | avg loss = %.6f | acc = %.4f\n",
                        epoch + 1, model.avg_loss(), model.accuracy());
            std::printf("  rank0 timing: compute = %.1f ms | comm span = %.1f ms | total = %.1f ms | overlap ~= %.1f ms | comm span %% = %.1f%%\n",
                        epoch_compute_ms, epoch_comm_ms, epoch_total_ms,
                        epoch_overlap_ms, comm_pct);
        }
    }

    CUDACHECK(cudaEventDestroy(iter_start));
    CUDACHECK(cudaEventDestroy(compute_end));
    CUDACHECK(cudaEventDestroy(comm_span_start));
    CUDACHECK(cudaEventDestroy(comm_span_end));
    CUDACHECK(cudaEventDestroy(sync_end));
    for (GradBucket &bucket : buckets)
    {
        CUDACHECK(cudaEventDestroy(bucket.ready_event));
    }
    CUDACHECK(cudaStreamDestroy(comm_stream));

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

    // Prefer one visible device per local rank, but also handle Slurm GPU
    // masks where each rank sees only its assigned device as logical GPU 0.
    int local_rank = 0;
    const char *local_rank_env = std::getenv("SLURM_LOCALID");
    if (local_rank_env != nullptr)
    {
        local_rank = std::atoi(local_rank_env);
    }
    int device_count = 0;
    CUDACHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0)
    {
        std::printf("Rank %d found no CUDA devices\n", rank);
        exit(EXIT_FAILURE);
    }

    int device = local_rank < device_count ? local_rank : 0;
    CUDACHECK(cudaSetDevice(device));
    // Warm up the lazy stream singleton so NCCL and kernels share it.
    (void)get_cuda_stream();

    ncclComm_t comm;
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

    const char data_dir[] = "data";
    const char output_path[] = "ff.params";

    // Default bucket_size keeps one bucket per layer. Smaller values split at
    // tensor boundaries where possible: e.g. ./train_dist 1024
    int bucket_size = (argc >= 2) ? std::atoi(argv[1]) : 25450;

    train(rank, size, data_dir, output_path, comm, bucket_size);

    ncclCommDestroy(comm);
    // Stream is owned by the get_cuda_stream() static and is released at exit.
    MPICHECK(MPI_Finalize());
    return 0;
}
