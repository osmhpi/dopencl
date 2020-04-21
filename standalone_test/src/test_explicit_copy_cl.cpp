/* Tests whether a a buffer shared in a context containing two OpenCL devices which
   is copied using an explicit copy (clEnqueueCopyBuffer) is successfully transferred */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

struct device_opencl
{
    cl::CommandQueue queue;
    cl::Buffer buf;
};

#define NUM_DEVICES 2

#ifndef NUM_COPIES
#define NUM_COPIES 1
#endif
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
#if 0 // For testing with a single device
    devices = std::vector<cl::Device>(NUM_DEVICES, devices[0]);
#endif
    if (devices.size() < NUM_DEVICES)
        throw std::runtime_error("Not enough OpenCL devices available");
    devices.resize(NUM_DEVICES);

    cl::Context context(devices);

    device_opencl devinfo[NUM_DEVICES];
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl &dev = devinfo[i];
        dev.queue = cl::CommandQueue(context, devices[i]);
        dev.buf = cl::Buffer(context, CL_MEM_READ_WRITE, BUF_SIZE);
    }

    // --------------
    // BUFFER FILLING
    // --------------
    auto start_time = std::chrono::steady_clock::now();

    // TODOXXX Why is this event necessary for dOpenCL and not on real HW (at least NVIDIA?)
    // Is this a bug in dOpenCL, or does NVIDIA use a stronger consistency model than that of the spec.?
    // See also: The tests/src/MemoryConsistency.cpp in the dOpenCL tree
    // See also: synchronize() method in daemon/src/CommandQueue.cpp on dOpenCL tree
    cl::Event event;

    // NB: Avoiding clEnqueueFillBuffer since it's not supported by dOpenCL
    std::vector<char> initial_buf(BUF_SIZE, 'A');
    devinfo[0].queue.enqueueWriteBuffer(devinfo[0].buf, CL_TRUE, 0,
                                        BUF_SIZE, initial_buf.data(), nullptr, &event);

    // -----------------------
    // CROSS-GPU DATA EXCHANGE
    // -----------------------
    for (cl_uint c = 0; c < NUM_COPIES; c++) {
        device_opencl &dev = devinfo[c%NUM_DEVICES], &nextDev = devinfo[(c+1)%NUM_DEVICES];

        std::vector<cl::Event> eventVector = {event};
        nextDev.queue.enqueueCopyBuffer(dev.buf, nextDev.buf,
                                        0, 0, BUF_SIZE, &eventVector, &event);
    }

    // --------------
    // GATHER RESULTS
    // --------------
    std::cout << "**AFTER COPY**\n";
    std::string buf_start(8, '*'), buf_end(8, '*');
    size_t lastDev = NUM_COPIES % NUM_DEVICES;
    devinfo[lastDev].queue.enqueueReadBuffer(devinfo[lastDev].buf, CL_TRUE,
                                             0, buf_start.size(), &buf_start[0]);
    devinfo[lastDev].queue.enqueueReadBuffer(devinfo[lastDev].buf, CL_TRUE,
                                             BUF_SIZE - buf_end.size(), buf_end.size(), &buf_end[0]);

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
