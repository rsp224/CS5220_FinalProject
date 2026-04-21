cmake_minimum_required(VERSION 3.18)

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif ()

project(DistributedNN VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

add_compile_options(-O3 -Wall -Wextra)

find_package(CUDAToolkit REQUIRED)

option(USE_MPI "Build distributed executable with MPI" OFF)

set(CRAY_CUDA_AWARE_MPI_TARGET "nvidia80" CACHE STRING
    "Cray accelerator target used to link the CUDA-aware MPI GTL library")

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message(WARNING
        "Recommended to use PrgEnv-gnu on Perlmutter for this project.")
endif ()

set(COMMON_SOURCES
    src/MNIST.cpp
    src/Model.cpp
    src/Stream.cpp
    src/cuda/Tensor.cu
    src/cuda/FFLayer.cu
    src/cuda/CCELoss.cu
    src/cuda/SGDOptimizer.cu
)

add_executable(train_single
    src/main.cpp
    ${COMMON_SOURCES}
)

target_include_directories(train_single PRIVATE include)

target_link_libraries(train_single PRIVATE
    CUDA::cudart
    CUDA::cublas
)

if (USE_MPI)
    find_package(MPI REQUIRED)

    if (NOT CMAKE_CXX_COMPILER MATCHES "(/|^)CC$")
        message(WARNING
            "Distributed GPU builds on Perlmutter are best configured with the Cray C++ wrapper `CC`.\n"
            "Recommended configure command:\n"
            "  module load cudatoolkit nccl craype-accel-nvidia80\n"
            "  cmake -S . -B build -DUSE_MPI=ON -DCMAKE_CXX_COMPILER=$(which CC)")
    endif ()

    set(NCCL_ROOT "" CACHE PATH "Root of the NCCL installation")
    set(_nccl_hints
        ${NCCL_ROOT}
        $ENV{NCCL_ROOT}
        $ENV{NCCL_DIR}
        $ENV{NCCL_HOME}
    )
    list(FILTER _nccl_hints EXCLUDE REGEX "^$")

    find_path(NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS ${_nccl_hints}
        PATH_SUFFIXES include
    )

    find_library(NCCL_LIBRARY
        NAMES nccl
        HINTS ${_nccl_hints}
        PATH_SUFFIXES lib lib64
    )

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(NCCL DEFAULT_MSG
        NCCL_INCLUDE_DIR NCCL_LIBRARY)

    add_library(NCCL::NCCL UNKNOWN IMPORTED)
    set_target_properties(NCCL::NCCL PROPERTIES
        IMPORTED_LOCATION "${NCCL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}"
    )

    add_executable(train_dist
        src/main_dist.cpp
        src/DistributedDataLoader.cpp
        ${COMMON_SOURCES}
    )

    target_include_directories(train_dist PRIVATE include)

    if (CMAKE_CXX_COMPILER MATCHES "(/|^)CC$")
        target_compile_options(train_dist PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-target-accel=${CRAY_CUDA_AWARE_MPI_TARGET}>
        )
        target_link_options(train_dist PRIVATE
            -target-accel=${CRAY_CUDA_AWARE_MPI_TARGET}
        )
    elseif (DEFINED ENV{PE_MPICH_GTL_DIR_nvidia80})
        string(REGEX REPLACE "^-L" "" MPICH_GTL_DIR "$ENV{PE_MPICH_GTL_DIR_nvidia80}")
        find_library(MPI_GTL_CUDA_LIBRARY
            NAMES mpi_gtl_cuda
            HINTS "${MPICH_GTL_DIR}"
        )
        if (MPI_GTL_CUDA_LIBRARY)
            target_link_libraries(train_dist PRIVATE "${MPI_GTL_CUDA_LIBRARY}")
        else ()
            message(WARNING
                "Could not find libmpi_gtl_cuda in ${MPICH_GTL_DIR}. "
                "If you hit a GTL runtime error, reconfigure with -DCMAKE_CXX_COMPILER=$(which CC).")
        endif ()
    endif ()

    target_link_libraries(train_dist PRIVATE
        CUDA::cudart
        CUDA::cublas
        MPI::MPI_CXX
        NCCL::NCCL
    )
endif ()