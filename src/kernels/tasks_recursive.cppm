// Partition :tasks_recursive — divide and conquer with explicit tasks.
//
// OpenMP concepts shown here:
//   * `parallel` + `single` — the usual task launch pattern: the team exists to
//                           execute tasks, one thread creates the root.
//   * `task`              — a unit of work the runtime may run on any thread
//                           of the team, now or later.
//   * `taskwait`          — wait for the child tasks of the current task.
//   * `taskgroup`         — wait for all tasks created inside it, including
//                           grandchildren.
//   * `if(expr)`          — when false the task is *undeferred*: run at once
//                           by the creating thread (still a task, data env.
//                           semantics kept).
//   * `final(expr)`       — when true all descendants are *included* tasks:
//                           executed immediately, removing task overhead deep
//                           in the recursion. omp_in_final() queries it.
//   * `mergeable`         — included tasks may share the parent's data env.
//   * `firstprivate` is the default for task variables that are not shared in
//     the enclosing context; we still spell it out with default(none).
//
// Splitting m or n gives two halves that write disjoint parts of C -> two
// tasks. Splitting k gives two products that add into the SAME C block, so
// they must run one after the other (doing them as concurrent tasks would be
// a data race).
module;

#include <omp.h>

export module matmul:tasks_recursive;

import std;
import :matrix;

namespace matmul {

struct Block {
    const double* a;
    const double* b;
    double* c;
    std::size_t m, n, k;       // C block is m x n, inner dimension k
    std::size_t lda, ldb, ldc; // leading dimensions (row strides)
};

// Below this many multiply-adds a block is computed directly.
inline constexpr std::size_t leaf_work = 32 * 32 * 32;
// Below this depth tasks are deferred; deeper ones run inline.
inline constexpr int final_depth = 12;

void leaf(const Block& blk) noexcept
{
    for (std::size_t i = 0; i < blk.m; ++i) {
        double* const crow = blk.c + i * blk.ldc;
        for (std::size_t p = 0; p < blk.k; ++p) {
            const double aip = blk.a[i * blk.lda + p];
            const double* const brow = blk.b + p * blk.ldb;
#pragma omp simd
            for (std::size_t j = 0; j < blk.n; ++j) {
                crow[j] += aip * brow[j];
            }
        }
    }
}

void recurse(Block blk, int depth);

// Runs two independent sub-products, as tasks unless we are already inside a
// final task (then creating tasks would only add overhead).
void run_pair(Block first, Block second, int depth, bool big)
{
    if (omp_in_final()) {
        recurse(first, depth + 1);
        recurse(second, depth + 1);
        return;
    }
#pragma omp task default(none) firstprivate(first, depth) if(big) final(depth >= final_depth) mergeable
    recurse(first, depth + 1);
#pragma omp task default(none) firstprivate(second, depth) if(big) final(depth >= final_depth) mergeable
    recurse(second, depth + 1);
#pragma omp taskwait
}

// C_block += A_block * B_block
void recurse(Block blk, int depth)
{
    if (blk.m * blk.n * blk.k <= leaf_work) {
        leaf(blk);
        return;
    }
    const bool big = blk.m * blk.n * blk.k > 64 * leaf_work;

    if (blk.m >= blk.n && blk.m >= blk.k) {
        const std::size_t h = blk.m / 2;
        Block top = blk;
        top.m = h;
        Block bottom = blk;
        bottom.m = blk.m - h;
        bottom.a += h * blk.lda;
        bottom.c += h * blk.ldc;
        run_pair(top, bottom, depth, big);
    } else if (blk.n >= blk.k) {
        const std::size_t h = blk.n / 2;
        Block left = blk;
        left.n = h;
        Block right = blk;
        right.n = blk.n - h;
        right.b += h;
        right.c += h;
        run_pair(left, right, depth, big);
    } else {
        // Split k: both halves accumulate into the same C block -> sequential.
        const std::size_t h = blk.k / 2;
        Block first = blk;
        first.k = h;
        Block second = blk;
        second.k = blk.k - h;
        second.a += h;
        second.b += h * blk.ldb;
        recurse(first, depth + 1);
        recurse(second, depth + 1);
    }
}

export void multiply_tasks_recursive(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    fill_parallel(c, 0.0);
    const Block root{a.data(), b.data(), c.data(), a.rows(), b.cols(), a.cols(), a.cols(), b.cols(), c.cols()};

#pragma omp parallel default(none) firstprivate(root)
#pragma omp single
    {
#pragma omp taskgroup
        recurse(root, 0);
        // Everything spawned (at any depth) has finished here.
    }
}

} // namespace matmul
