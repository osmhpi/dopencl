/* Tests for race condition between clEnqueueWriteBuffer transfers and node synchronization */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#ifndef BUF_SIZE
#define BUF_SIZE (40000)
#endif
#ifndef BUF2_SIZE
#define BUF2_SIZE (500)
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
    cl::CommandQueue queue(context, devices[0]);

    // --------------
    // BUFFER FILLING
    // --------------
    auto start_time = std::chrono::steady_clock::now();

    // TODOXXX Why is this event necessary for dOpenCL and not on real HW (at least NVIDIA?)
    // Is this a bug in dOpenCL, or does NVIDIA use a stronger consistency model than that of the spec.?
    // See also: The tests/src/MemoryConsistency.cpp in the dOpenCL tree
    // See also: synchronize() method in daemon/src/CommandQueue.cpp on dOpenCL tree
    std::vector<unsigned char> data1(40000, 'A');
    std::vector<unsigned char> data2(500, 'B');

    // Create some buffers
    const size_t NUM_TRIES = 10;
    std::vector<cl::Buffer> buffers(NUM_TRIES);
    for (size_t i = 0; i < NUM_TRIES; i++) {
        buffers[i] = cl::Buffer(context, static_cast<cl_mem_flags>(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), BUF_SIZE, data1.data());
    }

    // ---------
    // TRANSFERS
    // ---------
    // Do some enqueueWriteBuffer over those buffers. Each enqueueWriteBuffer will cause 2 transfers:
    // One to get the buffer to the node, and one to get the data to write to the node
    // The point of this test is to try to get those to interleave in an improper way
    // and trigger a race condition that crashes the program to crash, if that's possible
    std::vector<cl::Event> events(NUM_TRIES);
    for (size_t i = 0; i < NUM_TRIES; i++) {
        queue.enqueueWriteBuffer(buffers[i], CL_FALSE, 0, BUF2_SIZE, data2.data(), nullptr, &events[i]);
    }
    cl::Event::waitForEvents(events);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ----------------
    // VALIDATE RESULTS
    // ----------------
    auto test_is_ok = true; // This tests a crash condition so if it reaches this point, it has passed

    std::cout << "Time:     " << duration_ms << " ms\n";
    std::cout << "Result:   " << (test_is_ok ? "OK" : "KO") << "\n";

    return test_is_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
