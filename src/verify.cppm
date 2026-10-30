// Partition :verify — compare a kernel's result with the reference.
//
// OpenMP concepts shown here:
//   * `declare reduction` — a user-defined reduction over a struct, so one
//                           parallel loop finds both the max error and where
//                           it occurred.
//   * `cancel for` / `cancellation point for` — stop a parallel loop early as
//                           soon as any thread finds a mismatch. Only active
//                           when OMP_CANCELLATION=true; otherwise the loop
//                           simply runs to completion (still correct).
//   * `atomic write`      — publish a shared flag without a data race.
module;

#include <omp.h>

export module matmul:verify;

import std;
import :matrix;

namespace matmul {

export struct ErrorReport {
    double max_abs_error = 0.0;
    std::size_t index = 0;
};

[[nodiscard]] ErrorReport larger(const ErrorReport& x, const ErrorReport& y) noexcept
{
    return y.max_abs_error > x.max_abs_error ? y : x;
}

// omp_out / omp_in / omp_priv / omp_orig are the reserved names OpenMP uses
// for the combiner and initializer expressions.
#pragma omp declare reduction(max_error : ErrorReport : omp_out = larger(omp_out, omp_in)) \
    initializer(omp_priv = ErrorReport{})

export [[nodiscard]] ErrorReport max_error(const Mat& expected, const Mat& actual)
{
    if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
        throw std::invalid_argument{"max_error: shape mismatch"};
    }
    const double* const e = expected.data();
    const double* const g = actual.data();
    const std::size_t count = expected.size();

    ErrorReport report;
#pragma omp parallel for default(none) firstprivate(e, g, count) reduction(max_error : report)
    for (std::size_t idx = 0; idx < count; ++idx) {
        const double err = std::abs(e[idx] - g[idx]);
        // NaN-safe: a NaN result must count as an error.
        if (err > report.max_abs_error || std::isnan(err)) {
            report = {std::isnan(err) ? std::numeric_limits<double>::infinity() : err, idx};
        }
    }
    return report;
}

// True when every element is within `tolerance`. Bails out early via
// cancellation if the runtime has it enabled.
export [[nodiscard]] bool all_close(const Mat& expected, const Mat& actual, double tolerance)
{
    if (expected.rows() != actual.rows() || expected.cols() != actual.cols()) {
        return false;
    }
    const double* const e = expected.data();
    const double* const g = actual.data();
    const std::size_t count = expected.size();

    bool mismatch = false;
    // Separate `parallel` + `for` (not the combined construct): the combined
    // form is lowered as `for nowait`, where `cancel for` would have no effect.
#pragma omp parallel default(none) firstprivate(e, g, count, tolerance) shared(mismatch)
    {
#pragma omp for
        for (std::size_t idx = 0; idx < count; ++idx) {
            if (!(std::abs(e[idx] - g[idx]) <= tolerance)) {
#pragma omp atomic write
                mismatch = true;
#pragma omp cancel for
            }
            // Threads that did not hit `cancel` notice the request here.
#pragma omp cancellation point for
        }
    }
    return !mismatch;
}

// Reasonable bound for summation-order differences between kernels: each
// element is a dot product of length k with entries in [-1, 1].
export [[nodiscard]] constexpr double default_tolerance(std::size_t k) noexcept
{
    return 64.0 * std::numeric_limits<double>::epsilon() * static_cast<double>(std::max<std::size_t>(k, 1));
}

export [[nodiscard]] bool cancellation_enabled() noexcept { return omp_get_cancellation() != 0; }

} // namespace matmul
