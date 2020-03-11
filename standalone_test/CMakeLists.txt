cmake_minimum_required(VERSION 3.1)

project(dOpenCLstandalonetest)

find_package(TIFF)
include_directories(${TIFF_INCLUDE_DIRS})
link_directories(${TIFF_LIBRARY_DIRS})

foreach(test bitonic
             mapreduce_addition
             test_createbuffer_ptr
             test_createbuffer_ptr_multi
             test_createbuffer_ptr_race
             test_explicit_copy_cl
             test_implicit_copy_cl
             read_write_compressed
             text_search)
	add_executable(${test} src/${test}.cpp)
	target_link_libraries(${test} OpenCL)
	set_target_properties(${test} PROPERTIES COMPILE_DEFINITIONS "CL_HPP_TARGET_OPENCL_VERSION=120")
endforeach(test)

if (TIFF_FOUND)
	add_executable(downsize_tiff src/downsize_tiff.cpp)
	target_link_libraries(downsize_tiff OpenCL ${TIFF_LIBRARIES})
	set_target_properties(downsize_tiff PROPERTIES COMPILE_DEFINITIONS "CL_HPP_TARGET_OPENCL_VERSION=120")
endif()
