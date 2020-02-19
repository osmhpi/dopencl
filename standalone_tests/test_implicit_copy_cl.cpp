/* Tests whether a a buffer shared in a context containing two OpenCL devices which
   is copied using an implicit copy (clEnqueueNDRangeKernel) is successfully transferred */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

static const std::string OPENCL_PROGRAM = R"V0G0N(
__kernel void increment_each(__global char *buf, ulong buf_size)
{
    size_t x = get_global_id(0);

    //Make sure we do not go out of bounds
    if (x >= buf_size)
        return;

    buf[x]++;
}
)V0G0N";

typedef struct device_opencl
{
    cl::CommandQueue queue;
} device_opencl;

#define NUM_DEVICES 2
#define BUF_SIZE (1048576*30)

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
    cl::Program program(context, OPENCL_PROGRAM, true);
    cl::Kernel kernel(program, "increment_each");
    cl::Buffer buf(context, CL_MEM_READ_WRITE, BUF_SIZE);

    device_opencl devinfo[NUM_DEVICES];
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl &dev = devinfo[i];
        dev.queue = cl::CommandQueue(context, devices[i]);
    }

    // --------------
    // BUFFER FILLING
    // --------------
    // TODOXXX Why is this event necessary for dOpenCL and not on real HW (at least NVIDIA?)
    // Is this a bug in dOpenCL, or does NVIDIA use a stronger consistency model than that of the spec.?
    // See also: The tests/src/MemoryConsistency.cpp in the dOpenCL tree
    // See also: synchronize() method in daemon/src/CommandQueue.cpp on dOpenCL tree
    cl::Event event;

    // NB: Avoiding clEnqueueFillBuffer since it's not supported by dOpenCL
    std::vector<char> initial_buf(BUF_SIZE, 'A');
    devinfo[0].queue.enqueueWriteBuffer(buf, CL_TRUE, 0,
                                        BUF_SIZE, initial_buf.data(), nullptr, &event);

    // -------
    // KERNELS
    // -------
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl &dev = devinfo[(i+1)%NUM_DEVICES];
        kernel.setArg(0, buf);
        kernel.setArg(1, static_cast<cl_ulong>(BUF_SIZE));
        std::vector<cl::Event> eventVector = {event};
        cl::Event nextEvent;
        dev.queue.enqueueNDRangeKernel(kernel, cl::NullRange, BUF_SIZE, cl::NullRange,
                                       &eventVector, &nextEvent);
        event = nextEvent;
    }

    // ---------------------
    // PRINT BUFFER (AFTER)
    // ---------------------
    std::cout << "**AFTER KERNELS**\n";
    std::string buf_start(8, '*'), buf_end(8, '*');
    devinfo[0].queue.enqueueReadBuffer(buf, CL_TRUE,
                                       0, buf_start.size(), &buf_start[0]);
    devinfo[0].queue.enqueueReadBuffer(buf, CL_TRUE,
                                       BUF_SIZE - buf_end.size(), buf_end.size(), &buf_end[0]);
    std::cout << "Buffer: " << buf_start << "..." << buf_end << "\n";

    std::string expected_start(8, 'A' + NUM_DEVICES), expected_end(8, 'A' + NUM_DEVICES);
    std::cout << "Expected: " << expected_start << "..." << expected_end << "\n";

    return (buf_start == expected_start && buf_end == expected_end) ? EXIT_SUCCESS : EXIT_FAILURE;
}
