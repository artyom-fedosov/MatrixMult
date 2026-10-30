// Partition :matrix — the dense row-major matrix type shared by every kernel.
//
// OpenMP concepts shown here:
//   * `parallel sections`  — two independent pieces of work (filling A and
//                            filling B) handed to different threads.
//   * `parallel for simd`  — combined construct: threads split the loop and
//                            each thread vectorises its chunk.
//   * `aligned` clause     — promises the compiler a pointer alignment so it
//                            can emit aligned vector loads/stores.
//   * first-touch          — on NUMA machines a page lives on the node of the
//                            thread that first writes it, so initialising in
//                            parallel matters (see `fill_parallel`).
module;

#include <omp.h>

export module matmul:matrix;

import std;

namespace matmul {

// Cache line / AVX-512 friendly alignment for the matrix storage.
export inline constexpr std::size_t matrix_alignment = 64;

// Minimal over-aligned allocator so the `aligned` clause below is truthful.
template <typename T>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() = default;
    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U>&) noexcept {}

    [[nodiscard]] T* allocate(std::size_t count)
    {
        return static_cast<T*>(::operator new(count * sizeof(T), std::align_val_t{matrix_alignment}));
    }

    void deallocate(T* ptr, std::size_t) noexcept
    {
        ::operator delete(ptr, std::align_val_t{matrix_alignment});
    }

    friend bool operator==(const AlignedAllocator&, const AlignedAllocator&) = default;
};

export template <std::floating_point T>
class Matrix {
public:
    using value_type = T;
    using extents_type = std::dextents<std::size_t, 2>;

    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, T init = T{})
        : rows_{rows}, cols_{cols}, data_(rows * cols, init)
    {
    }

    [[nodiscard]] constexpr std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] constexpr std::size_t cols() const noexcept { return cols_; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return data_.size(); }

    // C++23 multidimensional subscript: m[i, j].
    [[nodiscard]] T& operator[](std::size_t i, std::size_t j) noexcept { return data_[i * cols_ + j]; }
    [[nodiscard]] const T& operator[](std::size_t i, std::size_t j) const noexcept { return data_[i * cols_ + j]; }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    [[nodiscard]] std::span<T> row(std::size_t i) noexcept { return {data() + i * cols_, cols_}; }
    [[nodiscard]] std::span<const T> row(std::size_t i) const noexcept { return {data() + i * cols_, cols_}; }

    [[nodiscard]] std::mdspan<T, extents_type> view() noexcept { return {data(), rows_, cols_}; }
    [[nodiscard]] std::mdspan<const T, extents_type> view() const noexcept { return {data(), rows_, cols_}; }

    // Parallel fill: every thread writes (and therefore first-touches) its own
    // contiguous chunk of the storage.
    void fill_parallel(T value) noexcept
    {
        T* const p = data();
        const std::size_t count = size();
#pragma omp parallel for simd default(none) firstprivate(p, count, value) aligned(p : matrix_alignment) schedule(static)
        for (std::size_t idx = 0; idx < count; ++idx) {
            p[idx] = value;
        }
    }

    friend bool operator==(const Matrix&, const Matrix&) = default;

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<T, AlignedAllocator<T>> data_;
};

export using Mat = Matrix<double>;

// Throws if C = A * B is not shape-consistent.
export void check_shapes(const Mat& a, const Mat& b, const Mat& c)
{
    if (a.cols() != b.rows() || c.rows() != a.rows() || c.cols() != b.cols()) {
        throw std::invalid_argument{std::format("shape mismatch: ({}x{}) * ({}x{}) -> ({}x{})",
                                                a.rows(), a.cols(), b.rows(), b.cols(), c.rows(), c.cols())};
    }
}

// Counter-based pseudo-random value in [-1, 1): the value depends only on
// (seed, index), never on which thread computes it, so results are identical
// for any thread count or schedule.
[[nodiscard]] constexpr double hash_to_unit(std::uint64_t seed, std::uint64_t index) noexcept
{
    std::uint64_t z = seed + 0x9e3779b97f4a7c15ULL * (index + 1);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-52 - 1.0;
}

void fill_random(Mat& m, std::uint64_t seed) noexcept
{
    double* const p = m.data();
    const std::size_t count = m.size();
    // Called from inside a `section`; this loop runs on the section's thread.
    for (std::size_t idx = 0; idx < count; ++idx) {
        p[idx] = hash_to_unit(seed, idx);
    }
}

export struct Inputs {
    Mat a;
    Mat b;
};

// Builds A (m x k) and B (k x n). The two fills are independent, so each one
// is a `section`: the team's threads pick sections up one at a time.
export [[nodiscard]] Inputs make_inputs(std::size_t m, std::size_t k, std::size_t n, std::uint64_t seed)
{
    Inputs in{Mat{m, k}, Mat{k, n}};
#pragma omp parallel sections default(none) shared(in) firstprivate(seed) num_threads(2)
    {
#pragma omp section
        fill_random(in.a, seed);
#pragma omp section
        fill_random(in.b, seed ^ 0xabcdef1234567890ULL);
    }
    return in;
}

} // namespace matmul
