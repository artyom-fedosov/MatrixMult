# MatrixMult — learning OpenMP through matrix multiplication

`C = A * B` implemented around twenty ways. Each way shows a different group of
OpenMP features. Every kernel is checked against a serial reference and
benchmarked by a small CLI. Written in C++23 with **modules** and
`import std;`, built with CMake and Ninja.

## Requirements

| Tool  | Version | Why |
|-------|---------|-----|
| GCC   | 15+ (tested with 16.2) | `import std;` from libstdc++, OpenMP 5.2 |
| CMake | 4.0+ (tested with 4.3) | C++ modules and the experimental `import std` support |
| Ninja | 1.11+  | the only CMake generator that supports C++ modules |

Clang works with modules too, but CMake supports `import std;` with Clang only
when it uses libc++.

## Build, test, run

```sh
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release

./build/gcc-release/matmul --list                 # kernels and the concepts they show
./build/gcc-release/matmul --size 1024            # benchmark all kernels
./build/gcc-release/matmul --shape 300 1000 200 --kernel tiled,task-depend
./build/gcc-release/matmul --kernel manual-spmd --verbose   # per-thread timing
./build/gcc-release/matmul --info                 # runtime settings + small demos
```

If configuring fails with an error about `CMAKE_EXPERIMENTAL_CXX_IMPORT_STD`,
your CMake release uses a different gate UUID. Find it in that release's
`Help/dev/experimental.rst` and pass `-DCMAKE_EXPERIMENTAL_CXX_IMPORT_STD=<uuid>`.

CMake options:

- `MATMUL_NATIVE` (ON): compile with `-march=native` in Release builds.
- `MATMUL_ENABLE_OFFLOAD` (OFF): when OFF, the project builds with
  `-foffload=disable`, so `target` regions always run on the host. Turn it ON
  if your GCC has offload support for a GPU installed.
- `MATMUL_BUILD_TESTS` (ON): build the test suite.

## Where each concept lives

| File | OpenMP concepts |
|------|-----------------|
| [`src/matrix.cppm`](src/matrix.cppm) | `parallel sections`, `parallel for simd`, `aligned`, first-touch placement |
| [`src/timer.cppm`](src/timer.cppm) | `omp_get_wtime`, `omp_get_wtick` |
| [`src/verify.cppm`](src/verify.cppm) | `declare reduction` (user-defined reduction on a struct), `cancel for`, `cancellation point`, `atomic write` |
| [`src/runtime.cppm`](src/runtime.cppm) | runtime query API, `threadprivate` + `copyin`, `single copyprivate`, `for ordered`, `masked`, named `critical`, nesting queries |
| [`src/kernels/serial.cppm`](src/kernels/serial.cppm) | baseline (no OpenMP) |
| [`src/kernels/parallel_for.cppm`](src/kernels/parallel_for.cppm) | `parallel for`, `default(none)`, `shared` / `private` / `firstprivate` / `lastprivate`, `parallel loop`, `loop bind(thread)` |
| [`src/kernels/collapse_schedule.cppm`](src/kernels/collapse_schedule.cppm) | `collapse(2)`, `schedule(static / dynamic / guided / runtime)`, `nonmonotonic`, `omp_set_schedule`, several worksharing loops in one region, `single` |
| [`src/kernels/simd.cppm`](src/kernels/simd.cppm) | `simd`, `simd reduction`, `declare simd` (`uniform`, `linear`, `notinbranch`), `simdlen`, `if(simd: …)` |
| [`src/kernels/tiled.cppm`](src/kernels/tiled.cppm) | cache blocking, `for nowait`, explicit `barrier`, `collapse` over tiles |
| [`src/kernels/manual_spmd.cppm`](src/kernels/manual_spmd.cppm) | bare `parallel` with thread ids, `atomic update`, `critical(name)`, `omp_lock_t` in an RAII wrapper, `masked`, avoiding false sharing |
| [`src/kernels/tasks_recursive.cppm`](src/kernels/tasks_recursive.cppm) | `task`, `taskwait`, `taskgroup`, `if` / `final` / `mergeable`, `omp_in_final` |
| [`src/kernels/taskloop.cppm`](src/kernels/taskloop.cppm) | `taskloop grainsize`, `taskloop num_tasks`, `nogroup`, `taskloop simd`, `masked` as the task producer |
| [`src/kernels/task_depend.cppm`](src/kernels/task_depend.cppm) | a task graph with `depend(in / out / inout)`, `priority` |
| [`src/kernels/nested.cppm`](src/kernels/nested.cppm) | nested parallelism, `num_threads`, `proc_bind(spread / close)`, max active levels |
| [`src/kernels/offload.cppm`](src/kernels/offload.cppm) | `target`, `map` with array sections, `target data`, `target update`, `teams distribute parallel for`, `begin/end declare target`, host `teams`, `omp_is_initial_device` |

Each file starts with a comment that explains its concepts and the usual
mistakes to avoid. A good reading order is the order of the table.

## Environment variables to experiment with

```sh
OMP_NUM_THREADS=2 ./build/gcc-release/matmul             # team size
OMP_SCHEDULE="static,32" ./build/gcc-release/matmul --kernel schedule-runtime
OMP_CANCELLATION=true ./build/gcc-release/matmul         # lets verification stop early
OMP_PLACES=cores OMP_PROC_BIND=spread,close ./build/gcc-release/matmul --kernel nested
OMP_MAX_ACTIVE_LEVELS=2 ./build/gcc-release/matmul --info
OMP_MAX_TASK_PRIORITY=100 ./build/gcc-release/matmul --kernel task-depend
OMP_DISPLAY_ENV=true ./build/gcc-release/matmul --list   # runtime prints its settings
```

The test suite runs the same checks with several of these settings (see
[`tests/CMakeLists.txt`](tests/CMakeLists.txt)).

## Sample results

These are from an i7-1260P with 4 threads, N = 1024, Release build. The
speedup column compares each kernel with the serial i-k-j reference.

```
kernel                 time [s]    GFLOP/s   speedup
serial                   0.2792       7.69     0.99x
parallel-for-ijk         2.1169       1.01     0.13x   textbook i-j-k: B read column-wise
parallel-for-ikj         0.1497      14.34     1.87x
collapse-static          0.3927       5.47     0.71x   dot product, no vectorised reduction
simd-dot                 0.1832      11.72     1.52x   same, with `simd reduction`
tiled                    0.1048      20.49     2.64x
tasks-recursive          0.1377      15.59     2.03x
taskloop-grainsize       0.1216      17.66     2.30x
task-depend              0.0806      26.65     3.44x   packed tiles + dependences
target-data              0.1267      16.94     2.20x   host fallback
```

Lessons from these numbers:

- **Memory access order matters more than thread count.** `parallel-for-ijk`
  on 4 threads is slower than the serial i-k-j loop.
- **Floating-point reductions only vectorise when you allow it.** Compare
  `collapse-static` with `simd-dot`.
- **Cache blocking and packing** (`tiled`, `task-depend`) are what move the
  numbers closer to the hardware peak.
- `target` regions without a GPU run on the host. The kernel is still correct,
  but loop orders chosen for a GPU can be slow on a CPU.

## Notes on modules + OpenMP

- `omp.h` is a C header, so it is included in the **global module fragment**
  (`module; #include <omp.h> export module …;`). `#pragma omp` works as usual
  inside the module.
- `_OPENMP` and the pragmas need `-fopenmp` on every translation unit,
  including the units that import the module. CMake handles this through
  `OpenMP::OpenMP_CXX`.
- With GCC, OpenMP clauses inside a **class-template member** that is
  instantiated in another module unit can lose their data-sharing information
  ("not specified in enclosing parallel"). For that reason the OpenMP code is
  kept in non-template functions, for example `fill_parallel`.
- GCC does not count a variable that appears only in a `depend` clause as
  used, so it may warn that the variable is unused. The code marks such
  variables `[[maybe_unused]]`.
