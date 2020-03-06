cmake_minimum_required(VERSION 2.6)

project(dOpenCLstandalonetest)

foreach(test bitonic
             mapreduce_addition
             test_createbuffer_ptr
             test_createbuffer_ptr_multi
             test_createbuffer_ptr_race
             test_explicit_copy_cl
             test_implicit_copy_cl)
	add_executable(${test} src/${test}.cpp)
	target_link_libraries(${test} OpenCL)
	set_target_properties(${test} PROPERTIES COMPILE_DEFINITIONS "CL_HPP_TARGET_OPENCL_VERSION=120")
endforeach(test)
