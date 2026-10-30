// Partition :manual_spmd — "Single Program, Multiple Data" by hand: a bare
// `parallel` region where each thread computes its own share from its id,
// plus the classic synchronisation primitives.
//
// OpenMP concepts shown here:
//   * `parallel` without worksharing, omp_get_thread_num / omp_get_num_threads
//   * `atomic update`     — lock-free increment of a shared counter.
//   * `critical(name)`    — named critical section; different names do not
//                           block each other.
//   * `omp_lock_t`        — runtime locks, wrapped in an RAII type that
//                           satisfies BasicLockable, so std::lock_guard works.
//   * `masked`            — the primary thread publishes the team size.
//   * false sharing       — per-thread data is padded to a cache line.
module;

#include <omp.h>

export module matmul:manual_spmd;

import std;
import :matrix;

namespace matmul {

// RAII owner of an OpenMP lock. lock()/unlock() make it usable with
// std::lock_guard / std::scoped_lock.
export class OmpLock {
public:
    OmpLock() noexcept { omp_init_lock(&lock_); }
    ~OmpLock() { omp_destroy_lock(&lock_); }
    OmpLock(const OmpLock&) = delete;
    OmpLock& operator=(const OmpLock&) = delete;

    void lock() noexcept { omp_set_lock(&lock_); }
    void unlock() noexcept { omp_unset_lock(&lock_); }
    [[nodiscard]] bool try_lock() noexcept { return omp_test_lock(&lock_) != 0; }

private:
    omp_lock_t lock_{};
};

export struct ThreadStat {
    int thread = 0;
    std::size_t first_row = 0;
    std::size_t rows = 0;
    double seconds = 0.0;
};

export struct SpmdStats {
    int team_size = 0;
    std::size_t rows_done = 0;
    double slowest_thread = 0.0;
    std::vector<ThreadStat> threads; // sorted by thread id
};

SpmdStats g_last_stats;

// Statistics of the most recent multiply_manual_spmd call.
export [[nodiscard]] const SpmdStats& spmd_last_stats() noexcept { return g_last_stats; }

// Splits [0, total) into `parts` nearly equal blocks and returns block `index`.
[[nodiscard]] constexpr std::pair<std::size_t, std::size_t> block_range(std::size_t total, int index, int parts) noexcept
{
    const auto p = static_cast<std::size_t>(parts);
    const auto t = static_cast<std::size_t>(index);
    const std::size_t base = total / p;
    const std::size_t extra = total % p;
    const std::size_t begin = t * base + std::min(t, extra);
    return {begin, begin + base + (t < extra ? 1 : 0)};
}

export void multiply_manual_spmd(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    SpmdStats stats;
    OmpLock stats_lock;

    // Per-thread padded counters: neighbouring threads' counters never share
    // a cache line, so updating them does not cause false sharing.
    struct alignas(64) PaddedCount {
        std::size_t value = 0;
    };
    std::vector<PaddedCount> per_thread(static_cast<std::size_t>(omp_get_max_threads()));

#pragma omp parallel default(none) shared(pa, pb, pc, stats, stats_lock, per_thread) firstprivate(m, n, k)
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();

#pragma omp masked
        stats.team_size = nthreads;

        const auto [begin, end] = block_range(m, tid, nthreads);
        const double start = omp_get_wtime();

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
            ++per_thread[static_cast<std::size_t>(tid)].value;

            // Shared counter: a plain `++` here would be a data race.
#pragma omp atomic update
            ++stats.rows_done;
        }

        const double elapsed = omp_get_wtime() - start;

        // Short reduction-like update guarded by a named critical section.
#pragma omp critical(spmd_slowest)
        stats.slowest_thread = std::max(stats.slowest_thread, elapsed);

        // Longer update (allocation inside) guarded by the RAII lock.
        {
            std::lock_guard guard{stats_lock};
            stats.threads.push_back({tid, begin, per_thread[static_cast<std::size_t>(tid)].value, elapsed});
        }
    }

    if (stats.rows_done != m) {
        throw std::logic_error{"manual-spmd: not every row was computed"};
    }
    std::ranges::sort(stats.threads, {}, &ThreadStat::thread);
    g_last_stats = std::move(stats);
}

} // namespace matmul
