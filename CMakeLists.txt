cmake_minimum_required(VERSION 3.18)
project(DistributedNN VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

add_compile_options(-O3 -Wall -Wextra)
set(CMAKE_CUDA_ARCHITECTURES "80")

find_package(CUDAToolkit REQUIRED)

option(USE_MPI "Build distributed executable with MPI" OFF)

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Recommended to use GNU on Perlmutter. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_CXX_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(train_single
    src/main.cpp
    src/MNIST.cpp
    src/Model.cpp
    src/cuda/Tensor.cu
    src/cuda/FFLayer.cu
    src/cuda/CCELoss.cu
    src/cuda/SGDOptimizer.cu
)

target_include_directories(train_single PUBLIC
    include
)

target_link_libraries(train_single PUBLIC
    CUDA::cudart
    CUDA::cublas
)

if (USE_MPI)
    find_package(MPI REQUIRED)

    add_executable(train_dist
        src/main_dist.cpp
        src/MNIST.cpp
        src/Model.cpp
        src/cuda/Tensor.cu
        src/cuda/FFLayer.cu
        src/cuda/CCELoss.cu
        src/cuda/SGDOptimizer.cu
    )

    target_include_directories(train_dist PUBLIC
        include
    )

    target_link_libraries(train_dist PUBLIC
        CUDA::cudart
        CUDA::cublas
        MPI::MPI_CXX
    )
endif ()