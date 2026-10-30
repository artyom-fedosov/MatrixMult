// Partition :kernels — the registry of every multiplication kernel, so the
// CLI and the tests can iterate over all of them uniformly.
export module matmul:kernels;

import std;
import :matrix;
import :serial;

namespace matmul {

export using KernelFn = void (*)(const Mat&, const Mat&, Mat&);

export struct Kernel {
    std::string_view name;
    KernelFn fn;
    std::string_view concepts;
};

export inline constexpr std::array kernels{
    Kernel{"serial", &multiply_serial, "baseline, no OpenMP"},
};

export [[nodiscard]] const Kernel* find_kernel(std::string_view name) noexcept
{
    const auto it = std::ranges::find(kernels, name, &Kernel::name);
    return it == kernels.end() ? nullptr : &*it;
}

} // namespace matmul
