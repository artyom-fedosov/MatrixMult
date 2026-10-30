// Checks every registered kernel against the serial reference on awkward
// shapes (1, odd sizes, sizes just past a tile, rectangular, degenerate),
// plus the helpers in :matrix and :verify.
//
// CTest runs this binary several times with different OMP_* environments, so
// thread-count- or schedule-dependent bugs show up.
#include <omp.h>

import std;
import matmul;

namespace {

int g_failures = 0;

void expect(bool condition, std::string_view what)
{
    if (!condition) {
        ++g_failures;
        std::println(std::cerr, "FAIL: {}", what);
    }
}

struct Shape {
    std::size_t m, k, n;
};

constexpr std::array shapes{
    Shape{1, 1, 1},     Shape{7, 7, 7},     Shape{63, 63, 63}, Shape{65, 65, 65},  Shape{128, 128, 128},
    Shape{257, 33, 129}, Shape{33, 257, 65}, Shape{70, 1, 130}, Shape{3, 200, 1},  Shape{1, 300, 300},
    Shape{0, 5, 5},     Shape{5, 0, 5},     Shape{5, 5, 0},
};

void test_kernels(std::string_view filter)
{
    for (const auto& [m, k, n] : shapes) {
        const auto [a, b] = matmul::make_inputs(m, k, n, 7);
        matmul::Mat reference{m, n};
        matmul::multiply_serial(a, b, reference);
        const double tolerance = matmul::default_tolerance(k);

        for (const auto& kernel : matmul::kernels) {
            if (!filter.empty() && kernel.name != filter) {
                continue;
            }
            // Pre-fill with garbage: kernels must overwrite, not accumulate.
            matmul::Mat c{m, n, 12345.0};
            try {
                kernel.fn(a, b, c);
            } catch (const std::exception& e) {
                expect(false, std::format("{} threw on {}x{}x{}: {}", kernel.name, m, k, n, e.what()));
                continue;
            }
            const auto report = matmul::max_error(reference, c);
            expect(matmul::all_close(reference, c, tolerance),
                   std::format("{} on {}x{}x{}: max error {:.3e} at {}", kernel.name, m, k, n, report.max_abs_error,
                               report.index));
        }
    }
}

void test_shape_check()
{
    const matmul::Mat a{3, 4};
    const matmul::Mat b{5, 2};
    matmul::Mat c{3, 2};
    for (const auto& kernel : matmul::kernels) {
        bool threw = false;
        try {
            kernel.fn(a, b, c);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, std::format("{} accepted mismatched shapes", kernel.name));
    }
}

void test_verify()
{
    matmul::Mat x{50, 50, 1.0};
    matmul::Mat y = x;
    expect(matmul::all_close(x, y, 0.0), "identical matrices compare equal");

    y[17, 23] = 1.5;
    y[40, 2] = 0.75;
    const auto report = matmul::max_error(x, y);
    expect(!matmul::all_close(x, y, 1e-3), "all_close detects a mismatch");
    expect(report.max_abs_error == 0.5 && report.index == 17 * 50 + 23, "declare reduction finds the largest error");

    y[0, 0] = std::numeric_limits<double>::quiet_NaN();
    expect(!matmul::all_close(x, y, 1.0), "NaN counts as a mismatch");
}

void test_inputs_deterministic()
{
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
    const auto one = matmul::make_inputs(40, 30, 20, 99);
    omp_set_num_threads(saved);
    const auto many = matmul::make_inputs(40, 30, 20, 99);
    expect(one.a == many.a && one.b == many.b, "inputs do not depend on thread count");
    expect(one.a != matmul::make_inputs(40, 30, 20, 100).a, "seed changes the inputs");
}

void test_fill_parallel()
{
    matmul::Mat x{33, 17};
    matmul::fill_parallel(x, 2.5);
    expect(std::ranges::all_of(std::span{x.data(), x.size()}, [](double v) { return v == 2.5; }), "fill_parallel");
    expect(reinterpret_cast<std::uintptr_t>(x.data()) % matmul::matrix_alignment == 0, "storage is aligned");
}

} // namespace

int main(int argc, char** argv)
{
    const std::string_view filter = argc > 1 ? argv[1] : "";
    std::println("threads = {}, cancellation = {}, kernels = {}", omp_get_max_threads(), matmul::cancellation_enabled(),
                 filter.empty() ? "all" : filter);

    test_kernels(filter);
    test_shape_check();
    test_verify();
    test_inputs_deterministic();
    test_fill_parallel();

    if (g_failures != 0) {
        std::println(std::cerr, "{} check(s) failed", g_failures);
        return 1;
    }
    std::println("all checks passed");
    return 0;
}
