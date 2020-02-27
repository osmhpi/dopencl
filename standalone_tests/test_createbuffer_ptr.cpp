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

    buf[x] = ((buf[x] - 'A' + 1) % 26) + 'A';
}
)V0G0N";

typedef struct device_opencl
{
    cl::CommandQueue queue;
} device_opencl;

#define NUM_DEVICES 2

#ifndef NUM_BOUNCES
#define NUM_BOUNCES 2
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
    cl::Program program(context, OPENCL_PROGRAM, true);
    cl::Kernel kernel(program, "increment_each");

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

    std::vector<unsigned char> orig(BUF_SIZE, 'A');
    cl::Buffer buf(context, static_cast<cl_mem_flags>(CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR), BUF_SIZE, orig.data());
    // As a bonus test, make sure that CL_MEM_COPY_HOST_PTR did really copy our pointer,
    // by deallocating the associated memory now. Note that std::vector<T>.clear() does not guarantee this
    // See: https://web.archive.org/web/20190201121515/http://www.cplusplus.com/reference/vector/vector/clear/
    std::vector<unsigned char>().swap(orig);

    // Note that cl::Buffer / clCreateBuffer with CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR
    // differs from e.g. clEnqueueCopyBuffer in that it can fill a buffer without being associated
    // with a command queue or producing an event. Instead, it conceptually works like a "broadcast"
    // to all devices in the context. In most implementations this is implemented by lazily copying
    // the data to the devices when they need it

    // -------
    // KERNELS
    // -------
    kernel.setArg(0, buf);
    kernel.setArg(1, static_cast<cl_ulong>(BUF_SIZE));

    for (cl_uint b = 0; b < NUM_BOUNCES; b++) {
        for (cl_uint i = 0; i < NUM_DEVICES; i++) {
            device_opencl &nextDev = devinfo[(i + 1) % NUM_DEVICES];
            std::vector<cl::Event> eventVector;
            if (b > 0 || i > 0)
                eventVector = {event};
            nextDev.queue.enqueueNDRangeKernel(kernel, cl::NullRange, BUF_SIZE, cl::NullRange,
                                               &eventVector, &event);
        }
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

    std::string expected_start(8, 'A' + (NUM_DEVICES * NUM_BOUNCES) % 26),
                expected_end(8, 'A' + (NUM_DEVICES * NUM_BOUNCES) % 26);
    std::cout << "Expected: " << expected_start << "..." << expected_end << "\n";

    return (buf_start == expected_start && buf_end == expected_end) ? EXIT_SUCCESS : EXIT_FAILURE;
}
