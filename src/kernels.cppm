// Partition :kernels — the registry of every multiplication kernel, so the
// CLI and the tests can iterate over all of them uniformly.
export module matmul:kernels;

import std;
import :matrix;
import :serial;
import :parallel_for;
import :collapse_schedule;

namespace matmul {

export using KernelFn = void (*)(const Mat&, const Mat&, Mat&);

export struct Kernel {
    std::string_view name;
    KernelFn fn;
    std::string_view concepts;
};

export inline constexpr std::array kernels{
    Kernel{"serial", &multiply_serial, "baseline, no OpenMP"},
    Kernel{"parallel-for-ijk", &multiply_parallel_for_ijk, "parallel for, default(none), shared/private/firstprivate"},
    Kernel{"parallel-for-ikj", &multiply_parallel_for_ikj, "parallel for, lastprivate, cache-friendly loop order"},
    Kernel{"loop-construct", &multiply_loop_construct, "parallel loop, loop bind(thread)"},
    Kernel{"collapse-static", &multiply_collapse_static, "collapse(2), schedule(static), single, multiple for in one region"},
    Kernel{"collapse-dynamic", &multiply_collapse_dynamic, "collapse(2), schedule(nonmonotonic:dynamic,64)"},
    Kernel{"collapse-guided", &multiply_collapse_guided, "collapse(2), schedule(guided,16)"},
    Kernel{"schedule-runtime", &multiply_schedule_runtime, "schedule(runtime) driven by OMP_SCHEDULE"},
    Kernel{"schedule-api", &multiply_schedule_api, "omp_get_schedule/omp_set_schedule + schedule(runtime)"},
};

export [[nodiscard]] const Kernel* find_kernel(std::string_view name) noexcept
{
    const auto it = std::ranges::find(kernels, name, &Kernel::name);
    return it == kernels.end() ? nullptr : &*it;
}

} // namespace matmul
