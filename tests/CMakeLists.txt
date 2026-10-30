add_executable(test_kernels test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE matmul_core)
target_compile_options(test_kernels PRIVATE -Wall -Wextra -Wpedantic)

# The same checks under different OpenMP environments.
add_test(NAME kernels_default COMMAND test_kernels)

add_test(NAME kernels_1_thread COMMAND test_kernels)
set_tests_properties(kernels_1_thread PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=1")

add_test(NAME kernels_3_threads COMMAND test_kernels)
set_tests_properties(kernels_3_threads PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=3")

add_test(NAME kernels_oversubscribed COMMAND test_kernels)
set_tests_properties(kernels_oversubscribed PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=16")

add_test(NAME kernels_cancellation_schedule COMMAND test_kernels)
set_tests_properties(kernels_cancellation_schedule PROPERTIES
    ENVIRONMENT "OMP_CANCELLATION=true;OMP_SCHEDULE=static,7")

add_test(NAME kernels_bound_places COMMAND test_kernels)
set_tests_properties(kernels_bound_places PROPERTIES
    ENVIRONMENT "OMP_PLACES=cores;OMP_PROC_BIND=spread,close;OMP_MAX_ACTIVE_LEVELS=2")

# Command line driver smoke tests.
add_test(NAME cli_run COMMAND matmul --size 96 --reps 1)
add_test(NAME cli_list COMMAND matmul --list)
add_test(NAME cli_info COMMAND matmul --info)
add_test(NAME cli_rejects_unknown_kernel COMMAND matmul --kernel nope)
set_tests_properties(cli_rejects_unknown_kernel PROPERTIES WILL_FAIL TRUE)
