// Primary module interface: `import matmul;` gives access to everything.
// Every interface partition must be re-exported from here.
export module matmul;

export import :matrix;
export import :timer;
export import :verify;
export import :runtime;
export import :kernels;
export import :serial;
