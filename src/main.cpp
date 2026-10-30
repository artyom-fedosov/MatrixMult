// Benchmark driver: runs the selected kernels on random matrices, checks each
// result against the serial reference and prints time, GFLOP/s and speedup.
#include <omp.h>

import std;
import matmul;

namespace {

struct Options {
    std::size_t m = 512;
    std::size_t k = 512;
    std::size_t n = 512;
    int reps = 3;
    int threads = 0; // 0 = leave the runtime default (OMP_NUM_THREADS)
    std::uint64_t seed = 42;
    std::vector<std::string_view> kernel_names;
    bool list = false;
    bool info = false;
    bool verbose = false;
};

void print_usage(std::string_view prog)
{
    std::println("Usage: {} [options]\n"
                 "  --size N            square matrices N x N (default 512)\n"
                 "  --shape M K N       C(MxN) = A(MxK) * B(KxN)\n"
                 "  --kernel NAME[,..]  kernels to run (default: all), repeatable\n"
                 "  --reps R            repetitions, best time is reported (default 3)\n"
                 "  --threads T         calls omp_set_num_threads(T)\n"
                 "  --seed S            input seed (default 42)\n"
                 "  --list              list kernels and the OpenMP concepts they show\n"
                 "  --info              print OpenMP runtime information and demos\n"
                 "  --verbose           per-thread statistics for kernels that record them\n"
                 "  --help              show this text",
                 prog);
}

template <typename T>
[[nodiscard]] T parse_number(std::string_view text, std::string_view what)
{
    T value{};
    const auto [ptr, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc{} || ptr != text.data() + text.size()) {
        throw std::invalid_argument{std::format("invalid value '{}' for {}", text, what)};
    }
    return value;
}

[[nodiscard]] std::optional<Options> parse_args(std::span<char* const> args)
{
    Options opt;
    std::size_t cursor = 1;
    auto take = [&](std::string_view flag) -> std::string_view {
        if (cursor >= args.size()) {
            throw std::invalid_argument{std::format("missing value for {}", flag)};
        }
        return args[cursor++];
    };

    while (cursor < args.size()) {
        const std::string_view flag = args[cursor++];
        if (flag == "--help" || flag == "-h") {
            print_usage(args[0]);
            return std::nullopt;
        } else if (flag == "--size") {
            opt.m = opt.k = opt.n = parse_number<std::size_t>(take(flag), flag);
        } else if (flag == "--shape") {
            opt.m = parse_number<std::size_t>(take(flag), flag);
            opt.k = parse_number<std::size_t>(take(flag), flag);
            opt.n = parse_number<std::size_t>(take(flag), flag);
        } else if (flag == "--kernel") {
            for (auto part : std::views::split(take(flag), ',')) {
                opt.kernel_names.emplace_back(part.begin(), part.end());
            }
        } else if (flag == "--reps") {
            opt.reps = parse_number<int>(take(flag), flag);
        } else if (flag == "--threads") {
            opt.threads = parse_number<int>(take(flag), flag);
        } else if (flag == "--seed") {
            opt.seed = parse_number<std::uint64_t>(take(flag), flag);
        } else if (flag == "--list") {
            opt.list = true;
        } else if (flag == "--info") {
            opt.info = true;
        } else if (flag == "--verbose" || flag == "-v") {
            opt.verbose = true;
        } else {
            throw std::invalid_argument{std::format("unknown option '{}'", flag)};
        }
    }
    return opt;
}

void print_spmd_stats()
{
    const auto& stats = matmul::spmd_last_stats();
    for (const auto& t : stats.threads) {
        std::println("    thread {:>2}: rows [{}, {}) in {:.4f} s", t.thread, t.first_row, t.first_row + t.rows, t.seconds);
    }
    std::println("    team {}, rows {}, slowest thread {:.4f} s", stats.team_size, stats.rows_done, stats.slowest_thread);
}

void list_kernels()
{
    std::println("{:<20} {}", "kernel", "OpenMP concepts");
    for (const auto& kernel : matmul::kernels) {
        std::println("{:<20} {}", kernel.name, kernel.concepts);
    }
}

[[nodiscard]] std::vector<const matmul::Kernel*> select_kernels(const Options& opt)
{
    std::vector<const matmul::Kernel*> selected;
    if (opt.kernel_names.empty() || std::ranges::contains(opt.kernel_names, "all")) {
        for (const auto& kernel : matmul::kernels) {
            selected.push_back(&kernel);
        }
        return selected;
    }
    for (const auto name : opt.kernel_names) {
        const auto* kernel = matmul::find_kernel(name);
        if (kernel == nullptr) {
            throw std::invalid_argument{std::format("unknown kernel '{}' (see --list)", name)};
        }
        selected.push_back(kernel);
    }
    return selected;
}

int run(const Options& opt)
{
    if (opt.threads > 0) {
        omp_set_num_threads(opt.threads);
    }
    if (opt.list) {
        list_kernels();
        return 0;
    }
    if (opt.info) {
        matmul::print_runtime_info();
        matmul::run_runtime_demos();
        return 0;
    }

    const auto kernels = select_kernels(opt);
    const auto [a, b] = matmul::make_inputs(opt.m, opt.k, opt.n, opt.seed);
    const double flops = 2.0 * static_cast<double>(opt.m) * static_cast<double>(opt.k) * static_cast<double>(opt.n);
    const double tolerance = matmul::default_tolerance(opt.k);

    std::println("C({}x{}) = A({}x{}) * B({}x{}), threads = {}, reps = {}, tolerance = {:.1e}",
                 opt.m, opt.n, opt.m, opt.k, opt.k, opt.n, omp_get_max_threads(), opt.reps, tolerance);

    matmul::Mat reference{opt.m, opt.n};
    const double serial_time = matmul::best_of(1, [&] { matmul::multiply_serial(a, b, reference); });

    std::println("{:<20} {:>10} {:>10} {:>9} {:>11}  {}", "kernel", "time [s]", "GFLOP/s", "speedup", "max error", "status");
    bool all_ok = true;
    for (const auto* kernel : kernels) {
        matmul::Mat c{opt.m, opt.n};
        double seconds = 0.0;
        try {
            seconds = matmul::best_of(opt.reps, [&] { kernel->fn(a, b, c); });
        } catch (const std::exception& e) {
            std::println("{:<20} failed: {}", kernel->name, e.what());
            all_ok = false;
            continue;
        }
        const auto report = matmul::max_error(reference, c);
        const bool ok = matmul::all_close(reference, c, tolerance);
        all_ok = all_ok && ok;
        std::println("{:<20} {:>10.4f} {:>10.2f} {:>8.2f}x {:>11.2e}  {}",
                     kernel->name, seconds, flops / seconds * 1e-9, serial_time / seconds,
                     report.max_abs_error, ok ? "ok" : "MISMATCH");
        if (opt.verbose && kernel->fn == &matmul::multiply_manual_spmd) {
            print_spmd_stats();
        }
    }
    return all_ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const auto opt = parse_args(std::span{argv, static_cast<std::size_t>(argc)});
        return opt ? run(*opt) : 0;
    } catch (const std::exception& e) {
        std::println(std::cerr, "error: {}", e.what());
        return 2;
    }
}
