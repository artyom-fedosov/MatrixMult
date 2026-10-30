// Partition :task_depend — a tile-level task graph driven by dependences.
//
// OpenMP concepts shown here:
//   * `depend(out: x)`    — this task produces x.
//   * `depend(in: x)`     — this task must wait for the last producer of x;
//                           several `in` tasks on the same x can run together.
//   * `depend(inout: x)`  — read-modify-write: serialised with every earlier
//                           task that names x.
//   * `priority(p)`       — a hint to run some ready tasks first (bounded by
//                           OMP_MAX_TASK_PRIORITY, 0 by default).
//
// Graph per C tile (ti, tj):
//   pack A(ti, tp) --in-->  multiply-accumulate (ti, tj, tp) --inout--> C(ti, tj)
//   pack B(tp, tj) --in-->            ...
//   zero C(ti, tj) --out-> first multiply on that tile
// The "dependence objects" are bytes in token arrays: OpenMP only compares
// addresses, so any distinct lvalue per tile works.
export module matmul:task_depend;

import std;
import :matrix;

namespace matmul {

inline constexpr std::size_t dep_tile = 64;

struct TileGrid {
    std::size_t rows, cols;
    [[nodiscard]] constexpr std::size_t index(std::size_t r, std::size_t c) const noexcept { return r * cols + c; }
};

[[nodiscard]] constexpr std::size_t tiles_for(std::size_t extent) noexcept { return (extent + dep_tile - 1) / dep_tile; }

// Copies a (rows x cols) block at (r0, c0) of a row-major matrix with leading
// dimension ld into a contiguous dep_tile x dep_tile buffer (zero padded).
void pack_tile(const double* src, std::size_t ld, std::size_t r0, std::size_t c0, std::size_t rows, std::size_t cols,
               double* dst) noexcept
{
    std::fill_n(dst, dep_tile * dep_tile, 0.0);
    for (std::size_t r = 0; r < rows; ++r) {
        std::copy_n(src + (r0 + r) * ld + c0, cols, dst + r * dep_tile);
    }
}

export void multiply_task_depend(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    const TileGrid ga{tiles_for(m), tiles_for(k)}; // tiles of A
    const TileGrid gb{tiles_for(k), tiles_for(n)}; // tiles of B
    const TileGrid gc{tiles_for(m), tiles_for(n)}; // tiles of C
    // Plain locals so they can be listed in data-sharing clauses.
    const std::size_t tile = dep_tile;
    const std::size_t tile_elems = tile * tile;

    std::vector<double> packed_a(ga.rows * ga.cols * tile_elems);
    std::vector<double> packed_b(gb.rows * gb.cols * tile_elems);
    std::vector<char> tok_a(ga.rows * ga.cols);
    std::vector<char> tok_b(gb.rows * gb.cols);
    std::vector<char> tok_c(gc.rows * gc.cols);
    double* const qa = packed_a.data();
    double* const qb = packed_b.data();
    // GCC does not count uses inside depend clauses, hence [[maybe_unused]].
    [[maybe_unused]] char* const ta = tok_a.data();
    [[maybe_unused]] char* const tb = tok_b.data();
    [[maybe_unused]] char* const tc = tok_c.data();

#pragma omp parallel default(none) shared(pa, pb, pc, qa, qb, ta, tb, tc) firstprivate(m, n, k, ga, gb, gc, tile, tile_elems)
#pragma omp single
    {
        // Producers: pack every tile of A and B.
        for (std::size_t r = 0; r < ga.rows; ++r) {
            for (std::size_t s = 0; s < ga.cols; ++s) {
                const std::size_t r0 = r * tile;
                const std::size_t c0 = s * tile;
                const std::size_t rows = std::min(tile, m - r0);
                const std::size_t cols = std::min(tile, k - c0);
                double* const dst = qa + ga.index(r, s) * tile_elems;
#pragma omp task default(none) shared(pa, ta) firstprivate(r0, c0, rows, cols, dst, k, r, s, ga) \
    depend(out : ta[ga.index(r, s)])
                pack_tile(pa, k, r0, c0, rows, cols, dst);
            }
        }
        for (std::size_t r = 0; r < gb.rows; ++r) {
            for (std::size_t s = 0; s < gb.cols; ++s) {
                const std::size_t r0 = r * tile;
                const std::size_t c0 = s * tile;
                const std::size_t rows = std::min(tile, k - r0);
                const std::size_t cols = std::min(tile, n - c0);
                double* const dst = qb + gb.index(r, s) * tile_elems;
#pragma omp task default(none) shared(pb, tb) firstprivate(r0, c0, rows, cols, dst, n, r, s, gb) \
    depend(out : tb[gb.index(r, s)])
                pack_tile(pb, n, r0, c0, rows, cols, dst);
            }
        }

        for (std::size_t ti = 0; ti < gc.rows; ++ti) {
            for (std::size_t tj = 0; tj < gc.cols; ++tj) {
                const std::size_t rows = std::min(tile, m - ti * tile);
                const std::size_t cols = std::min(tile, n - tj * tile);
                double* const ctile = pc + ti * tile * n + tj * tile;

#pragma omp task default(none) firstprivate(ctile, rows, cols, n) shared(tc) firstprivate(ti, tj, gc) \
    depend(out : tc[gc.index(ti, tj)])
                for (std::size_t r = 0; r < rows; ++r) {
                    std::fill_n(ctile + r * n, cols, 0.0);
                }

                for (std::size_t tp = 0; tp < ga.cols; ++tp) {
                    const double* const atile = qa + ga.index(ti, tp) * tile_elems;
                    const double* const btile = qb + gb.index(tp, tj) * tile_elems;
                    // Earlier k-steps get a higher priority so tile chains
                    // advance evenly.
                    const int prio = static_cast<int>(ga.cols - tp);
#pragma omp task default(none) firstprivate(atile, btile, ctile, rows, cols, n, tile) shared(ta, tb, tc) \
    firstprivate(ti, tj, tp, ga, gb, gc) priority(prio)                                             \
    depend(in : ta[ga.index(ti, tp)], tb[gb.index(tp, tj)]) depend(inout : tc[gc.index(ti, tj)])
                    for (std::size_t r = 0; r < rows; ++r) {
                        double* const crow = ctile + r * n;
                        for (std::size_t p = 0; p < tile; ++p) {
                            const double aip = atile[r * tile + p];
                            const double* const brow = btile + p * tile;
#pragma omp simd
                            for (std::size_t j = 0; j < cols; ++j) {
                                crow[j] += aip * brow[j];
                            }
                        }
                    }
                }
            }
        }
        // The implicit barrier at the end of `single`/`parallel` waits for
        // all tasks.
    }
}

} // namespace matmul
