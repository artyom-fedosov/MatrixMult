// Primary module interface: `import matmul;` gives access to everything.
// Every interface partition must be re-exported from here.
export module matmul;

export import :matrix;
export import :timer;
export import :verify;
export import :runtime;
export import :kernels;
export import :serial;
export import :parallel_for;
export import :collapse_schedule;
export import :simd;
export import :tiled;
export import :manual_spmd;
export import :tasks_recursive;
export import :taskloop;
export import :task_depend;
export import :nested;
export import :offload;
