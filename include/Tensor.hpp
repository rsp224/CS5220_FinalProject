#pragma once

#include <cstddef>
#include <driver_types.h>
#include <vector>

class Tensor {
public:
    Tensor();
    Tensor(int rows, int cols);
    ~Tensor();

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    void resize(int rows, int cols);
    void resize_like(const Tensor& other);

    void zero();

    void copy_from_host(const std::vector<float>& host_data);
    std::vector<float> copy_to_host() const;

    void copy_from_device(const Tensor& other);
    void copy_from_device(const Tensor& other, cudaStream_t stream);

    float* data() noexcept;
    const float* data() const noexcept;

    int rows() const noexcept;
    int cols() const noexcept;
    std::size_t size() const noexcept;
    std::size_t bytes() const noexcept;

    void div(float scalar);
    void div(float scalar, cudaStream_t stream);
    void accumulate_slice(const Tensor& other,
                          std::size_t dst_offset,
                          std::size_t src_offset,
                          std::size_t count,
                          cudaStream_t stream);

private:
    void free();

    float* data_;
    int rows_;
    int cols_;
    std::size_t size_;
};
