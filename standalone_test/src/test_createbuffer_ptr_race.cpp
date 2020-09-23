/* Tests for race condition between clEnqueueWriteBuffer transfers and node synchronization */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <chrono>

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
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.size() < 1)
        throw std::runtime_error("Not enough OpenCL devices available");
    devices.resize(1);

    cl::Context context(devices);
    cl::CommandQueue queue(context, devices[0]);

    // --------------
    // BUFFER FILLING
    // --------------
    auto start_time = std::chrono::steady_clock::now();

    // A note about events: Events are technically necessary in OpenCL for proper
    // synchronization of buffers on different command queues. This is documented
    // in the OpenCL 1.2 specification in Appendix A, 'Shared OpenCL Objects'.
    //
    // The most important point is: When enqueuing a command A accessing a buffer
    // last modified by a command B on another command queue, the event corresponding
    // to that command B must be provided in the event wait list of the command A
    //
    // However, it appears that in practice, a lot of implementations (such as NVIDIA's)
    // implement additional 'automatic' synchronization between command queues in
    // multi-GPU set-ups, so that synchonrization often happens even without events
    //
    // However, dOpenCL does not implement such extensions, and thus proper use
    // of events is required for synchronization
    //
    // See also: OpenCL 1.2 specification in Appendix A, 'Shared OpenCL Objects'.
    //           The tests/src/MemoryConsistency.cpp in the dOpenCL tree
    //           synchronize() method in daemon/src/CommandQueue.cpp on dOpenCL tree
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
