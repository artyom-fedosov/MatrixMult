// Partition :serial — the single-threaded reference every other kernel is
// checked against. No OpenMP here on purpose: it is the baseline for speedup.
//
// Loop order is i-p-j ("ikj"): the innermost loop walks a row of B and a row
// of C contiguously, which is far more cache friendly than the textbook i-j-k.
export module matmul:serial;

import std;
import :matrix;

namespace matmul {

export void multiply_serial(const Mat& a, const Mat& b, Mat& c)
{
    check_shapes(a, b, c);
    const std::size_t m = a.rows();
    const std::size_t k = a.cols();
    const std::size_t n = b.cols();

    std::ranges::fill(std::span{c.data(), c.size()}, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t p = 0; p < k; ++p) {
            const double aip = a[i, p];
            for (std::size_t j = 0; j < n; ++j) {
                c[i, j] += aip * b[p, j];
            }
        }
    }
}

} // namespace matmul
