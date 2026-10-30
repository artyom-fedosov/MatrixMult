// Partition :tiled — cache blocking with explicit synchronisation control.
//
// OpenMP concepts shown here:
//   * `for nowait`        — drop the implicit barrier at the end of a loop.
//   * `barrier`           — explicit team-wide synchronisation point. Here it
//                           restores the ordering `nowait` removed: every tile
//                           of C must be zeroed before any thread accumulates.
//                           (Each thread zeroes rows it may not later update,
//                           so the barrier is required.)
//   * `for collapse(2)` over tiles — each (ii, jj) tile of C is owned by one
//                           thread, so accumulation needs no locking.
//   * `simd`              — inner loop vectorised.
//
// Blocking keeps a TILE x TILE piece of A, B and C in cache while it is
// reused TILE times, which is what makes this kernel fast for large N.
export module matmul:tiled;

import std;
import :matrix;

namespace matmul {

export inline constexpr std::size_t tile_size = 64;

export void multiply_tiled(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();
    const std::size_t tile = tile_size;

#pragma omp parallel default(none) shared(pa, pb, pc) firstprivate(m, n, k, tile)
    {
#pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < m; ++i) {
            std::fill_n(pc + i * n, n, 0.0);
        }

#pragma omp barrier

#pragma omp for collapse(2) schedule(static)
        for (std::size_t ii = 0; ii < m; ii += tile) {
            for (std::size_t jj = 0; jj < n; jj += tile) {
                const std::size_t i_end = std::min(ii + tile, m);
                const std::size_t j_end = std::min(jj + tile, n);
                for (std::size_t pp = 0; pp < k; pp += tile) {
                    const std::size_t p_end = std::min(pp + tile, k);
                    for (std::size_t i = ii; i < i_end; ++i) {
                        double* const crow = pc + i * n;
                        for (std::size_t p = pp; p < p_end; ++p) {
                            const double aip = pa[i * k + p];
                            const double* const brow = pb + p * n;
#pragma omp simd
                            for (std::size_t j = jj; j < j_end; ++j) {
                                crow[j] += aip * brow[j];
                            }
                        }
                    }
                }
            }
        }
    }
}

} // namespace matmul
