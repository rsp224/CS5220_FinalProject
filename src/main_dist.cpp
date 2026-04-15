#include <mpi.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#define MPICHECK(cmd)                                                          \
  do {                                                                         \
    int e = cmd;                                                               \
    if (e != MPI_SUCCESS) {                                                    \
      printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,            \
             cudaGetErrorString(e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t r = cmd;                                                      \
    if (r != ncclSuccess) {                                                    \
      printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,            \
             ncclGetErrorString(r));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  int rank, size;

  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

  ncclUniqueId id;
  if (rank == 0)
    NCCLCHECK(ncclGetUniqueId(&id));
  MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  cudaStream_t stream;
  // Slurm should already handle GPU assignment where it assigns one GPU per
  // rank
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaStreamCreate(&stream));

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));

  printf("hello from rank %d\n", rank);

  ncclCommDestroy(comm);
  CUDACHECK(cudaStreamDestroy(stream));
  MPICHECK(MPI_Finalize());
  return 0;
}