/* Tests whether a a buffer shared in a context containing two OpenCL devices which
   is copied using an implicit copy (clEnqueueNDRangeKernel) is successfully transferred */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

static const std::string OPENCL_PROGRAM = R"V0G0N(
__kernel void add_buffers(__global const char *buf1,
                          __global const char *buf2,
                          __global const char *buf3,
                          __global const char *buf4,
                          __global const char *buf5,
                          __global char *out, ulong buf_size)
{
    size_t x = get_global_id(0);

    //Make sure we do not go out of bounds
    if (x >= buf_size)
        return;

    out[x] = buf1[x] + buf2[x/2] + buf3[x/3] + buf4[x/4] + buf5[x/5];
}
)V0G0N";

#ifndef BUF_SIZE
#define BUF_SIZE (1048576*30)
#endif

int main(void)
{
    // ---------------------
    // OPENCL INITIALIZATION
    // ---------------------
    cl::Platform platform;
    cl::Platform::get(&platform);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() < 1)
        throw std::runtime_error("Not enough OpenCL devices available");
    devices.resize(1);

    cl::Context context(devices);
    cl::Program program(context, OPENCL_PROGRAM, true);
    cl::Kernel kernel(program, "add_buffers");
    cl::CommandQueue queue(context, devices[0]);

    // --------------
    // BUFFER FILLING
    // --------------
    auto start_time = std::chrono::steady_clock::now();

    // TODOXXX Why is this event necessary for dOpenCL and not on real HW (at least NVIDIA?)
    // Is this a bug in dOpenCL, or does NVIDIA use a stronger consistency model than that of the spec.?
    // See also: The tests/src/MemoryConsistency.cpp in the dOpenCL tree
    // See also: synchronize() method in daemon/src/CommandQueue.cpp on dOpenCL tree
    cl::Event event;

    std::vector<unsigned char> orig(BUF_SIZE, 11);
    cl::Buffer buf1(context, static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), BUF_SIZE, orig.data());
    orig = std::vector<unsigned char>(BUF_SIZE/2, 12);
    cl::Buffer buf2(context, static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), BUF_SIZE / 2, orig.data());
    orig = std::vector<unsigned char>(BUF_SIZE/3, 13);
    cl::Buffer buf3(context, static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), BUF_SIZE / 3, orig.data());
    orig = std::vector<unsigned char>(BUF_SIZE/4, 14);
    cl::Buffer buf4(context, static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), BUF_SIZE / 4, orig.data());
    orig = std::vector<unsigned char>(BUF_SIZE/5, 15);
    cl::Buffer buf5(context, static_cast<cl_mem_flags>(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR), BUF_SIZE / 5, orig.data());

    cl::Buffer result(context, static_cast<cl_mem_flags>(CL_MEM_WRITE_ONLY), BUF_SIZE);

    // Note that cl::Buffer / clCreateBuffer with CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR
    // differs from e.g. clEnqueueCopyBuffer in that it can fill a buffer without being associated
    // with a command queue or producing an event. Instead, it conceptually works like a "broadcast"
    // to all devices in the context. In most implementations this is implemented by lazily copying
    // the data to the devices when they need it

    // -------
    // KERNELS
    // -------
    kernel.setArg(0, buf1);
    kernel.setArg(1, buf2);
    kernel.setArg(2, buf3);
    kernel.setArg(3, buf4);
    kernel.setArg(4, buf5);
    kernel.setArg(5, result);
    kernel.setArg(6, static_cast<cl_ulong>(BUF_SIZE));

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, BUF_SIZE, cl::NullRange, nullptr, &event);

    // --------------
    // GATHER RESULTS
    // --------------
    std::cout << "**AFTER KERNELS**\n";
    std::string buf_start(8, '*'), buf_end(8, '*');
    queue.enqueueReadBuffer(result, CL_TRUE, 0, buf_start.size(), &buf_start[0]);
    queue.enqueueReadBuffer(result, CL_TRUE, BUF_SIZE - buf_end.size(), buf_end.size(), &buf_end[0]);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ----------------
    // VALIDATE RESULTS
    // ----------------
    std::string expected_start(8, 'A'), expected_end(8, 'A');
    auto test_is_ok = buf_start == expected_start && buf_end == expected_end;

    std::cout << "Buffer:   " << buf_start << "..." << buf_end << "\n";
    std::cout << "Expected: " << expected_start << "..." << expected_end << "\n";
    std::cout << "Time:     " << duration_ms << " ms\n";
    std::cout << "Result:   " << (test_is_ok ? "OK" : "KO") << "\n";

    return test_is_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
