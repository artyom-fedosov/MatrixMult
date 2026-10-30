// Partition :runtime — querying the OpenMP runtime and small demos of the
// data-environment and synchronisation constructs that do not fit naturally
// into a multiplication kernel.
//
// OpenMP concepts shown here:
//   * runtime API: omp_get_num_procs, omp_get_max_threads, omp_get_schedule,
//     omp_get_max_active_levels, omp_get_proc_bind, omp_get_num_devices, ...
//   * `threadprivate` + `copyin`  — a global variable with one copy per thread
//                                   that persists between parallel regions.
//   * `single copyprivate`        — one thread produces a value, every thread
//                                   receives it in its private variable.
//   * `for ordered` / `ordered`   — run part of a parallel loop in sequential
//                                   iteration order.
//   * `masked`                    — only the primary thread (or a filtered one)
//                                   executes the block, no implied barrier.
//   * `critical(name)`            — mutual exclusion, here to keep output lines
//                                   from interleaving.
module;

#include <omp.h>

export module matmul:runtime;

import std;

namespace matmul {

// One copy per thread. Must be a variable with static storage duration.
int tp_counter = 0;
#pragma omp threadprivate(tp_counter)

[[nodiscard]] std::string_view schedule_name(omp_sched_t kind) noexcept
{
    // The monotonic modifier may be OR-ed into the kind.
    switch (static_cast<omp_sched_t>(kind & ~omp_sched_monotonic)) {
    case omp_sched_static: return "static";
    case omp_sched_dynamic: return "dynamic";
    case omp_sched_guided: return "guided";
    case omp_sched_auto: return "auto";
    default: return "implementation-defined";
    }
}

[[nodiscard]] std::string_view proc_bind_name(omp_proc_bind_t bind) noexcept
{
    switch (bind) {
    case omp_proc_bind_false: return "false";
    case omp_proc_bind_true: return "true";
    case omp_proc_bind_primary: return "primary";
    case omp_proc_bind_close: return "close";
    case omp_proc_bind_spread: return "spread";
    default: return "unknown";
    }
}

export void print_runtime_info()
{
    omp_sched_t kind{};
    int chunk = 0;
    omp_get_schedule(&kind, &chunk);

    std::println("OpenMP spec version (_OPENMP) : {}", _OPENMP);
    std::println("processors                    : {}", omp_get_num_procs());
    std::println("max threads                   : {}", omp_get_max_threads());
    std::println("thread limit                  : {}", omp_get_thread_limit());
    std::println("dynamic adjustment            : {}", omp_get_dynamic() != 0);
    std::println("max active levels             : {}", omp_get_max_active_levels());
    std::println("runtime schedule              : {}, chunk {}", schedule_name(kind), chunk);
    std::println("proc bind                     : {}", proc_bind_name(omp_get_proc_bind()));
    std::println("places                        : {}", omp_get_num_places());
    std::println("cancellation                  : {}", omp_get_cancellation() != 0);
    std::println("offload devices               : {}", omp_get_num_devices());
    std::println("max task priority             : {}", omp_get_max_task_priority());
    std::println("wtime resolution              : {:.3e} s", omp_get_wtick());
    std::println("in parallel / level           : {} / {}", omp_in_parallel() != 0, omp_get_level());
}

void demo_threadprivate()
{
    std::println("\n-- threadprivate + copyin --");
    tp_counter = 100; // sets the primary thread's copy

    // copyin: every thread's copy starts with the primary thread's value.
#pragma omp parallel copyin(tp_counter)
    {
        tp_counter += omp_get_thread_num();
    }
    // A second region: each thread still sees the value it left behind
    // (guaranteed when dynamic threads are off and the team size is unchanged).
#pragma omp parallel
    {
#pragma omp critical(print)
        std::println("thread {} kept tp_counter = {}", omp_get_thread_num(), tp_counter);
    }
}

void demo_copyprivate()
{
    std::println("\n-- single copyprivate --");
#pragma omp parallel
    {
        int token = -1; // private: declared inside the region
#pragma omp single copyprivate(token)
        {
            token = 1000 + omp_get_thread_num(); // produced by one thread
        }
        // The implied barrier at the end of `single` broadcasts `token`.
#pragma omp critical(print)
        std::println("thread {} received token {}", omp_get_thread_num(), token);
    }
}

void demo_ordered()
{
    std::println("\n-- for ordered + masked --");
#pragma omp parallel
    {
#pragma omp masked
        std::println("team of {} threads (printed by thread {} only)", omp_get_num_threads(), omp_get_thread_num());

        // Make sure the line above is printed before the loop output.
#pragma omp barrier

#pragma omp for ordered schedule(dynamic)
        for (int i = 0; i < 8; ++i) {
            const int square = i * i; // this part runs in parallel
#pragma omp ordered
            std::println("iteration {} (square {}) done by thread {}", i, square, omp_get_thread_num());
        }
    }
}

void demo_nesting_queries()
{
    std::println("\n-- nesting queries --");
    const int saved_levels = omp_get_max_active_levels();
    omp_set_max_active_levels(2);
#pragma omp parallel num_threads(2)
    {
#pragma omp parallel num_threads(2)
        {
#pragma omp critical(print)
            std::println("level {} (active {}), thread {} of {}, parent thread {}",
                         omp_get_level(), omp_get_active_level(), omp_get_thread_num(),
                         omp_get_num_threads(), omp_get_ancestor_thread_num(1));
        }
    }
    omp_set_max_active_levels(saved_levels);
}

export void run_runtime_demos()
{
    demo_threadprivate();
    demo_copyprivate();
    demo_ordered();
    demo_nesting_queries();
}

} // namespace matmul
