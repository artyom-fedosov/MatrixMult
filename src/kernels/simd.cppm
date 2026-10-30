// Partition :simd — vectorisation directives, combined with threading.
//
// OpenMP concepts shown here:
//   * `simd`                    — vectorise a loop: one thread, several lanes.
//   * `simd reduction(+:sum)`   — lets the compiler keep partial sums in vector
//                                 lanes. Without it, strict IEEE ordering of
//                                 `sum +=` forbids vectorising the dot product
//                                 (compare with the collapse-* kernels).
//   * `declare simd`            — compile a vector variant of a function so it
//                                 can be called from inside `simd` loops.
//                                 `uniform` = same value in every lane,
//                                 `linear` = increases with the lane index,
//                                 `notinbranch` = never called conditionally.
//   * `parallel for simd` is the combined form; it is used in
//     fill_parallel (matrix.cppm). It is a poor fit for the
//     multiplication itself: applied to a collapsed (i, j) space the lanes need
//     div/mod index decoding and gathers, and runs ~10x slower than simd-ikj.
//   * `simdlen`, `if(simd: ...)` — preferred vector length; vectorise only when
//                                 the loop is long enough to pay off.
export module matmul:simd;

import std;
import :matrix;

namespace matmul {

// y + alpha * x[i]; the vector variant receives `alpha` once and a pointer
// that advances by one element per lane.
#pragma omp declare simd uniform(alpha, x) linear(i : 1) notinbranch
[[nodiscard]] inline double axpy_at(double alpha, const double* x, std::size_t i, double y) noexcept
{
    return y + alpha * x[i];
}

// i-p-j order, threads over rows, vector lanes over columns.
export void multiply_simd_ikj(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

#pragma omp parallel for default(none) shared(pa, pb, pc) firstprivate(m, n, k) schedule(static)
    for (std::size_t i = 0; i < m; ++i) {
        double* const crow = pc + i * n;
#pragma omp simd
        for (std::size_t j = 0; j < n; ++j) {
            crow[j] = 0.0;
        }
        for (std::size_t p = 0; p < k; ++p) {
            const double aip = pa[i * k + p];
            const double* const brow = pb + p * n;
#pragma omp simd if(simd : n >= 8)
            for (std::size_t j = 0; j < n; ++j) {
                crow[j] = axpy_at(aip, brow, j, crow[j]);
            }
        }
    }
}

// Dot-product form on a transposed B with a vectorised reduction.
export void multiply_simd_dot(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    std::vector<double> bt(k * n);
    double* const pbt = bt.data();

#pragma omp parallel default(none) shared(pa, pb, pc, pbt) firstprivate(m, n, k)
    {
#pragma omp for collapse(2)
        for (std::size_t p = 0; p < k; ++p) {
            for (std::size_t j = 0; j < n; ++j) {
                pbt[j * k + p] = pb[p * n + j];
            }
        }

#pragma omp for collapse(2) schedule(static)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                const double* const arow = pa + i * k;
                const double* const bcol = pbt + j * k;
                double sum = 0.0;
#pragma omp simd reduction(+ : sum) simdlen(8)
                for (std::size_t p = 0; p < k; ++p) {
                    sum += arow[p] * bcol[p];
                }
                pc[i * n + j] = sum;
            }
        }
    }
}

} // namespace matmul
