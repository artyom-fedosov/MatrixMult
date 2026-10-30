// Partition :collapse_schedule — loop collapsing and loop scheduling.
//
// OpenMP concepts shown here:
//   * `collapse(2)`   — fuse the i and j loops into one iteration space of
//                       m*n iterations, so there is enough parallel work even
//                       when m is smaller than the number of threads.
//   * `schedule(static)`         — equal contiguous chunks, decided up front.
//   * `schedule(dynamic, chunk)` — threads grab chunks from a queue at run time
//                                  (helps with uneven iterations, costs overhead).
//   * `schedule(guided)`         — like dynamic but chunks shrink over time.
//   * `schedule(runtime)` + omp_set_schedule / OMP_SCHEDULE — choose without
//                                  recompiling.
//   * `nonmonotonic` modifier    — allows work stealing for dynamic schedules.
//   * one `parallel` region with several worksharing loops — avoids paying the
//     team start-up cost twice; the implicit barrier at the end of the first
//     `for` makes the transposed B visible before the second loop reads it.
//   * `single`        — one thread allocates the shared scratch buffer.
//
// The dot-product form C[i][j] = sum_p A[i][p] * B[p][j] needs B column-wise,
// so B is transposed first to keep the inner loop contiguous.
module;

#include <omp.h>

export module matmul:collapse_schedule;

import std;
import :matrix;

namespace matmul {

enum class Schedule { static_, dynamic_, guided_, runtime_ };

template <Schedule S>
void multiply_collapsed(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    std::vector<double> bt; // B transposed: bt[j * k + p] == B[p][j]

#pragma omp parallel default(none) shared(pa, pb, pc, bt) firstprivate(m, n, k)
    {
        // Exactly one thread resizes; the implicit barrier at the end of
        // `single` guarantees everyone sees the allocated buffer afterwards.
#pragma omp single
        bt.resize(k * n);

        double* const pbt = bt.data();
#pragma omp for collapse(2) schedule(static)
        for (std::size_t p = 0; p < k; ++p) {
            for (std::size_t j = 0; j < n; ++j) {
                pbt[j * k + p] = pb[p * n + j];
            }
        }
        // <- implicit barrier: transpose complete.

        // `if constexpr` picks one of several otherwise identical loops,
        // because the schedule kind must be spelled out in the pragma.
        if constexpr (S == Schedule::static_) {
#pragma omp for collapse(2) schedule(static)
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t p = 0; p < k; ++p) {
                        sum += pa[i * k + p] * pbt[j * k + p];
                    }
                    pc[i * n + j] = sum;
                }
            }
        } else if constexpr (S == Schedule::dynamic_) {
#pragma omp for collapse(2) schedule(nonmonotonic : dynamic, 64)
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t p = 0; p < k; ++p) {
                        sum += pa[i * k + p] * pbt[j * k + p];
                    }
                    pc[i * n + j] = sum;
                }
            }
        } else if constexpr (S == Schedule::guided_) {
#pragma omp for collapse(2) schedule(guided, 16)
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t p = 0; p < k; ++p) {
                        sum += pa[i * k + p] * pbt[j * k + p];
                    }
                    pc[i * n + j] = sum;
                }
            }
        } else {
#pragma omp for collapse(2) schedule(runtime)
            for (std::size_t i = 0; i < m; ++i) {
                for (std::size_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (std::size_t p = 0; p < k; ++p) {
                        sum += pa[i * k + p] * pbt[j * k + p];
                    }
                    pc[i * n + j] = sum;
                }
            }
        }
    }
}

export void multiply_collapse_static(const Mat& a, const Mat& b, Mat& c)
{
    multiply_collapsed<Schedule::static_>(a, b, c);
}

export void multiply_collapse_dynamic(const Mat& a, const Mat& b, Mat& c)
{
    multiply_collapsed<Schedule::dynamic_>(a, b, c);
}

export void multiply_collapse_guided(const Mat& a, const Mat& b, Mat& c)
{
    multiply_collapsed<Schedule::guided_>(a, b, c);
}

// Uses whatever OMP_SCHEDULE says (libgomp defaults to dynamic,1).
// Try: OMP_SCHEDULE="static,32" ./matmul --kernel schedule-runtime
export void multiply_schedule_runtime(const Mat& a, const Mat& b, Mat& c)
{
    multiply_collapsed<Schedule::runtime_>(a, b, c);
}

// Same, but chooses the schedule programmatically via omp_set_schedule.
export void multiply_schedule_api(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c); // throw before touching the global schedule
    omp_sched_t saved_kind{};
    int saved_chunk = 0;
    omp_get_schedule(&saved_kind, &saved_chunk);

    omp_set_schedule(omp_sched_guided, 8);
    multiply_collapsed<Schedule::runtime_>(a, b, c);

    omp_set_schedule(saved_kind, saved_chunk);
}

} // namespace matmul
