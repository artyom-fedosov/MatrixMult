// Partition :timer — wall-clock timing with the OpenMP runtime clock.
//
// OpenMP concepts shown here:
//   * omp_get_wtime() — portable, per-thread-consistent wall clock in seconds.
//   * omp_get_wtick() — resolution of that clock.
module;

#include <omp.h>

export module matmul:timer;

import std;

namespace matmul {

export [[nodiscard]] double wtime() noexcept { return omp_get_wtime(); }
export [[nodiscard]] double wtick() noexcept { return omp_get_wtick(); }

// RAII timer: stores the elapsed seconds into `out` when it goes out of scope.
export class ScopedTimer {
public:
    explicit ScopedTimer(double& out) noexcept : out_{out}, start_{omp_get_wtime()} {}
    ~ScopedTimer() { out_ = omp_get_wtime() - start_; }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    double& out_;
    double start_;
};

// Runs `fn` `reps` times and returns the fastest run in seconds.
export template <std::invocable F>
[[nodiscard]] double best_of(int reps, F&& fn)
{
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < std::max(reps, 1); ++r) {
        double elapsed = 0.0;
        {
            ScopedTimer timer{elapsed};
            std::invoke(fn);
        }
        best = std::min(best, elapsed);
    }
    return best;
}

} // namespace matmul
