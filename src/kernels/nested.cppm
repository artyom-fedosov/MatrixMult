// Partition :nested — a parallel region inside a parallel region.
//
// OpenMP concepts shown here:
//   * nested parallelism  — only active when max-active-levels > 1
//                           (omp_set_max_active_levels / OMP_MAX_ACTIVE_LEVELS);
//                           otherwise inner regions run with one thread.
//   * `num_threads(n)`    — team size for one specific region.
//   * `proc_bind(spread)` — spread the outer threads across the places
//                           (e.g. sockets/cores) ...
//   * `proc_bind(close)`  — ... and keep each inner team near its parent.
//                           Binding only has an effect with OMP_PLACES set,
//                           e.g. OMP_PLACES=cores.
//   * omp_get_level / omp_get_ancestor_thread_num — see runtime.cppm.
//
// Here the outer team splits the rows into blocks; each outer thread then
// starts an inner team that splits its block again. On a single socket this
// is not faster than one flat team — the point is the mechanics.
module;

#include <omp.h>

export module matmul:nested;

import std;
import :matrix;

namespace matmul {

// Restores the max-active-levels setting on scope exit.
class ActiveLevelsGuard {
public:
    explicit ActiveLevelsGuard(int levels) noexcept : saved_{omp_get_max_active_levels()}
    {
        omp_set_max_active_levels(levels);
    }
    ~ActiveLevelsGuard() { omp_set_max_active_levels(saved_); }
    ActiveLevelsGuard(const ActiveLevelsGuard&) = delete;
    ActiveLevelsGuard& operator=(const ActiveLevelsGuard&) = delete;

private:
    int saved_;
};

export void multiply_nested(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    const int total = omp_get_max_threads();
    const int outer = total >= 4 ? 2 : 1;
    const int inner = std::max(1, total / outer);

    const ActiveLevelsGuard levels{2};

#pragma omp parallel num_threads(outer) proc_bind(spread) default(none) shared(pa, pb, pc) firstprivate(m, n, k, inner)
    {
        const auto team = static_cast<std::size_t>(omp_get_num_threads());
        const auto me = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t begin = m * me / team;
        const std::size_t end = m * (me + 1) / team;

#pragma omp parallel for num_threads(inner) proc_bind(close) default(none) shared(pa, pb, pc) \
    firstprivate(begin, end, n, k) schedule(static)
        for (std::size_t i = begin; i < end; ++i) {
            double* const crow = pc + i * n;
            std::fill_n(crow, n, 0.0);
            for (std::size_t p = 0; p < k; ++p) {
                const double aip = pa[i * k + p];
                const double* const brow = pb + p * n;
#pragma omp simd
                for (std::size_t j = 0; j < n; ++j) {
                    crow[j] += aip * brow[j];
                }
            }
        }
    }
}

} // namespace matmul
