// Partition :taskloop — loops split into tasks instead of worksharing chunks.
//
// OpenMP concepts shown here:
//   * `taskloop grainsize(g)` — each task gets about g iterations.
//   * `taskloop num_tasks(t)` — create about t tasks in total.
//   * `nogroup`               — no implicit taskgroup around the taskloop; we
//                               then wait explicitly with `taskwait`.
//   * `taskloop simd`         — tasks whose chunks are also vectorised (used
//                               for zeroing C; on the multiplication itself
//                               the lanes would need strided gathers).
//   * `masked` as task producer — unlike `single`, `masked` has no implied
//                               barrier, so the other threads go straight to
//                               the barrier at the end of the parallel region
//                               and execute tasks while waiting there.
//
// Tasks shine when work is irregular or when loops are nested inside other
// tasks; for a regular loop like this, `for` is usually a bit cheaper.
module;

#include <omp.h>

export module matmul:taskloop;

import std;
import :matrix;

namespace matmul {

// C row i += sum_p A[i][p] * B[p][:]  (row must be zeroed beforehand)
inline void accumulate_row(const double* pa, const double* pb, double* pc, std::size_t i, std::size_t n, std::size_t k) noexcept
{
    double* const crow = pc + i * n;
    for (std::size_t p = 0; p < k; ++p) {
        const double aip = pa[i * k + p];
        const double* const brow = pb + p * n;
#pragma omp simd
        for (std::size_t j = 0; j < n; ++j) {
            crow[j] += aip * brow[j];
        }
    }
}

export void multiply_taskloop_grainsize(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

#pragma omp parallel default(none) shared(pa, pb, pc) firstprivate(m, n, k)
#pragma omp single
#pragma omp taskloop grainsize(8) default(none) shared(pa, pb, pc) firstprivate(m, n, k)
    for (std::size_t i = 0; i < m; ++i) {
        std::fill_n(pc + i * n, n, 0.0);
        accumulate_row(pa, pb, pc, i, n, k);
    }
}

export void multiply_taskloop_numtasks(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

#pragma omp parallel default(none) shared(pa, pb, pc) firstprivate(m, n, k)
    {
#pragma omp masked
        {
            // Zero C with vectorised tasks. The implicit taskgroup of this
            // taskloop guarantees C is zero before the next one starts.
            const std::size_t count = m * n;
#pragma omp taskloop simd grainsize(4096) default(none) shared(pc) firstprivate(count)
            for (std::size_t idx = 0; idx < count; ++idx) {
                pc[idx] = 0.0;
            }

            const int tasks = 4 * omp_get_num_threads();
#pragma omp taskloop num_tasks(tasks) nogroup default(none) shared(pa, pb, pc) firstprivate(m, n, k)
            for (std::size_t i = 0; i < m; ++i) {
                accumulate_row(pa, pb, pc, i, n, k);
            }
            // Without the implicit taskgroup, wait for the children ourselves.
#pragma omp taskwait
        }
    }
}

} // namespace matmul
