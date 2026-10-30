// Partition :parallel_for — the first step from serial to parallel.
//
// OpenMP concepts shown here:
//   * `parallel for`      — create a team and split the outer loop among it.
//   * `default(none)`     — forces every variable's data-sharing attribute to
//                           be stated explicitly; the best habit to catch races
//                           at compile time.
//   * `shared`, `private`, `firstprivate`, `lastprivate` — data-sharing clauses.
//   * `loop` construct    — OpenMP 5.0 descriptive loop: tells the compiler the
//                           iterations are independent and lets it choose how
//                           to parallelise them.
//
// Pitfall: parallelising the *k* loop instead of *i* would make every thread
// update the same C[i][j] — a data race. Splitting rows of C means each
// thread writes a disjoint set of elements, so no synchronisation is needed.
export module matmul:parallel_for;

import std;
import :matrix;

namespace matmul {

// Textbook i-j-k order with a private accumulator per (i, j).
export void multiply_parallel_for_ijk(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    // The loop variable `i` of the parallelised loop is private automatically.
    // `j` and `p` are declared outside the loop just to show `private`: each
    // thread gets its own *uninitialised* copy.
    std::size_t j = 0;
    std::size_t p = 0;
#pragma omp parallel for default(none) shared(pa, pb, pc) firstprivate(m, n, k) private(j, p) schedule(static)
    for (std::size_t i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            double sum = 0.0; // declared inside the region -> private
            for (p = 0; p < k; ++p) {
                sum += pa[i * k + p] * pb[p * n + j];
            }
            pc[i * n + j] = sum;
        }
    }
}

// Cache-friendly i-p-j order. Also demonstrates `lastprivate`: after the loop
// `last_row` holds the value from the sequentially last iteration (i = m - 1),
// whichever thread executed it.
export void multiply_parallel_for_ikj(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

    std::size_t last_row = 0;
#pragma omp parallel for default(none) shared(pa, pb, pc) firstprivate(m, n, k) lastprivate(last_row)
    for (std::size_t i = 0; i < m; ++i) {
        double* const crow = pc + i * n;
        std::fill_n(crow, n, 0.0);
        for (std::size_t p = 0; p < k; ++p) {
            const double aip = pa[i * k + p];
            const double* const brow = pb + p * n;
            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += aip * brow[j];
            }
        }
        last_row = i;
    }
    if (m > 0 && last_row != m - 1) {
        throw std::logic_error{"lastprivate did not deliver the last iteration"};
    }
}

// Same computation expressed with the descriptive `loop` construct.
// `bind(thread)` on the inner loop means "each thread runs it alone".
export void multiply_loop_construct(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();
    const double* const pa = a.data();
    const double* const pb = b.data();
    double* const pc = c.data();

#pragma omp parallel loop default(none) shared(pa, pb, pc) firstprivate(m, n, k)
    for (std::size_t i = 0; i < m; ++i) {
        double* const crow = pc + i * n;
        for (std::size_t j = 0; j < n; ++j) {
            crow[j] = 0.0;
        }
        for (std::size_t p = 0; p < k; ++p) {
            const double aip = pa[i * k + p];
            const double* const brow = pb + p * n;
#pragma omp loop bind(thread)
            for (std::size_t j = 0; j < n; ++j) {
                crow[j] += aip * brow[j];
            }
        }
    }
}

} // namespace matmul
