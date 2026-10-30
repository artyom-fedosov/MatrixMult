// Partition :offload — the device model: `target`, `teams`, `distribute`.
//
// OpenMP concepts shown here:
//   * `target`            — run a region on a device (GPU). With no device
//                           (or built with -foffload=disable, the default
//                           here) it falls back to the host: same results.
//   * `map(to: / from: / tofrom:)` with array sections `p[0:len]` — copy data
//                           between host and device memory.
//   * `target data`       — keep data on the device across several target
//                           regions instead of copying for each one.
//   * `target update`     — refresh a mapped variable explicitly.
//   * `teams distribute`  — a league of teams, iterations split between teams;
//                           combined with `parallel for` inside each team.
//   * `begin/end declare target` — compile a function for the device too.
//   * host `teams` (OpenMP 5.0) — teams on the CPU, e.g. one per NUMA node.
//   * omp_get_num_devices, omp_is_initial_device, omp_get_team_num.
module;

#include <omp.h>

export module matmul:offload;

import std;
import :matrix;

namespace matmul {

#pragma omp begin declare target
// Dot product of row i of A with column j of B. Callable inside target regions.
inline double dot_row_col(const double* pa, const double* pb, std::size_t i, std::size_t j, std::size_t n,
                          std::size_t k) noexcept
{
    double sum = 0.0;
    for (std::size_t p = 0; p < k; ++p) {
        sum += pa[i * k + p] * pb[p * n + j];
    }
    return sum;
}
#pragma omp end declare target

// True if target regions execute on the host (no device or offload disabled).
export [[nodiscard]] bool target_runs_on_host() noexcept
{
    int on_host = 1;
#pragma omp target map(from : on_host)
    on_host = omp_is_initial_device();
    return on_host != 0;
}

// One self-contained target region: map inputs in, result out.
export void multiply_target(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

#pragma omp target teams distribute parallel for collapse(2) map(to : pa[0 : m * k], pb[0 : k * n]) \
    map(from : pc[0 : m * n]) firstprivate(m, n, k)
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            pc[i * n + j] = dot_row_col(pa, pb, i, j, n, k);
        }
    }
}

// Two target regions sharing one device data environment.
export void multiply_target_data(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    double checksum_row0 = 0.0;

#pragma omp target data map(to : pa[0 : m * k], pb[0 : k * n]) map(from : pc[0 : m * n])
    {
        // Region 1: zero C on the device (no transfer: C is already mapped).
#pragma omp target teams distribute parallel for simd firstprivate(m, n)
        for (std::size_t idx = 0; idx < m * n; ++idx) {
            pc[idx] = 0.0;
        }

        // Region 2: rows split over teams and their threads, columns over
        // SIMD lanes. C is still on the device from region 1.
#pragma omp target teams distribute parallel for firstprivate(m, n, k)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t p = 0; p < k; ++p) {
                const double aip = pa[i * k + p];
#pragma omp simd
                for (std::size_t j = 0; j < n; ++j) {
                    pc[i * n + j] += aip * pb[p * n + j];
                }
            }
        }

        // Peek at the first row while the data region is still open.
        if (m > 0) {
#pragma omp target update from(pc[0 : n])
            for (std::size_t j = 0; j < n; ++j) {
                checksum_row0 += pc[j];
            }
        }
    }
    // map(from:) copied all of C back at the end of `target data`.
    if (m > 0) {
        double again = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            again += pc[j];
        }
        if (again != checksum_row0) {
            throw std::logic_error{"target update and final map(from:) disagree"};
        }
    }
}

// Host teams: a league of CPU teams, rows distributed between them, each team
// running its own parallel loop with at most thread_limit threads.
export void multiply_teams_host(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    const int total = omp_get_max_threads();
    const int teams = total >= 2 ? 2 : 1;
    const int per_team = std::max(1, total / teams);

#pragma omp teams num_teams(teams) thread_limit(per_team) default(none) shared(pa, pb, pc) firstprivate(m, n, k)
#pragma omp distribute parallel for dist_schedule(static) schedule(static)
    for (std::size_t i = 0; i < m; ++i) {
        double* const crow = pc + i * n;
        for (std::size_t j = 0; j < n; ++j) {
            crow[j] = 0.0;
        }
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

} // namespace matmul
