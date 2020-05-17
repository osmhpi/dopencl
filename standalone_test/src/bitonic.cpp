/* Tests sorting a large array (possibly on multiple GPUs) using Bitonic Sort */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <queue>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

static const std::string OPENCL_PROGRAM = R"V0G0N(
__kernel void bitonic_sort(__global ushort *array, ulong size,
                           ulong phase, ulong subphase,
                           ulong global_base_idx)
{
    ulong k = get_global_id(0);
    if (k >= size)
        return;

    // Calculate the distance to the element we possibly have to swap with
    ulong stride = 1 << (phase - subphase);

    // We actually waste half of the kernel invocations here;
    // With a smarter implementation, we could avoid this
    if ((k & stride) != 0)
        return;

    // Calculate swap direction; Note that for multi-device execution,
    // we need global_base_idx to determine the right swap direction
    bool direction = (((k + global_base_idx) >> phase) & 2) != 0;

    if ((array[k] > array[k + stride]) ^ direction) {
        ushort temp = array[k];
        array[k] = array[k + stride];
        array[k + stride] = temp;
    }
}
)V0G0N";

struct device_opencl
{
    cl::CommandQueue queue;
    cl::Buffer buf;
};

#define NUM_DEVICES 2
#define BUFFER_SIZE 1048576

static size_t log2(size_t x) {
    assert(x != 0 && (x & (x - 1)) == 0); // x is a power of two
    size_t logX = 0;
    for (size_t k = 1; k < x; k *= 2) {
        logX++;
    }
    return logX;
}

int main(void) {
    // ----------
    // DATA SETUP
    // ----------
    std::vector<std::uint16_t> array(BUFFER_SIZE);

    unsigned xorshift_seed = 123456;
    for (auto &v : array) {
        v = static_cast<std::uint16_t>(xorshift_seed & 0xffff);
        xorshift_seed ^= xorshift_seed << 13;
        xorshift_seed ^= xorshift_seed >> 17;
        xorshift_seed ^= xorshift_seed << 5;
    }

    std::vector<std::uint16_t> expectedArray = array;
    std::sort(expectedArray.begin(), expectedArray.end());

    // ---------------------
    // OPENCL INITIALIZATION
    // ---------------------
    cl::Platform platform;
    cl::Platform::get(&platform);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
#if 0 // For testing with a single device
    devices = std::vector<cl::Device>(NUM_DEVICES, devices[0]);
#endif
    if (devices.size() < NUM_DEVICES)
        throw std::runtime_error("Not enough OpenCL devices available");
    devices.resize(NUM_DEVICES);

    cl::Context context(devices);
    cl::Program program(context, OPENCL_PROGRAM, true);
    cl::Kernel kernel(program, "bitonic_sort");
    device_opencl devinfo[NUM_DEVICES];
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl &dev = devinfo[i];
        dev.queue = cl::CommandQueue(context, devices[i]);
        dev.buf = cl::Buffer(context, CL_MEM_READ_WRITE, array.size() * sizeof(array[0]));
    }

    // -------------------------
    // COPY DATA HOST -> DEVICES
    // -------------------------
    auto start_time = std::chrono::steady_clock::now();

    std::vector<cl::Event> events(NUM_DEVICES);

    for (size_t d = 0; d < NUM_DEVICES; d++) {
        devinfo[d].queue.enqueueWriteBuffer(devinfo[d].buf, CL_FALSE,
            0, array.size() / NUM_DEVICES * sizeof(array[0]),
            array.data() + d * array.size() / NUM_DEVICES,
            nullptr, &events[d]);
    }

    // -------
    // KERNELS
    // -------
    /* When only 1 device is used, Bitonic sort is implemented as usual.
     *
     * When 2 devices are used:
     * * The data split so that each device receives half of the data.
     * * It can be easily seen by the diagrams in 
     *   https://en.wikipedia.org/wiki/Bitonic_sorter, that all phases and
     *   sub-phases except the first sub-phase of the last phase only require
     *   in-device swaps. Therefore, they can be executed without communication
     *   with the host or other devices.
     *   (however, one must be careful that in the next-to-last phase,
          the swap direction is inverted in both devices!)
     * * In first sub-phase of the last phase, all data is gathered on one
     *   device, and then executed as in the 1 device case. Afterwards,
     *   the data is split again on both devices.
     *
     * It can easily be seen that this strategy generalizes to any number of
     * devices that is a power of two, however, actually implementing
     * it is a bit tricky.
     * 
     */
    size_t log2N = log2(array.size());

    for (size_t phase = 0; phase < log2N; phase++) {
        static_assert(NUM_DEVICES == 1 || NUM_DEVICES == 2, "Only 1 or 2 devices are supported");

        for (size_t subphase = 0; subphase <= phase; subphase++) {
            //std::cout << "phase=" << phase << ", subphase=" << subphase << std::endl;

            bool needGatherScatter = NUM_DEVICES == 2 && phase == log2N - 1 && subphase == 0;

            if (needGatherScatter) {
                // Gather all 
                std::vector<cl::Event> eventVector = {events[0], events[1]};

                devinfo[0].queue.enqueueCopyBuffer(
                    devinfo[1].buf, devinfo[0].buf,
                    0, array.size() / 2 * sizeof(array[0]),
                    array.size() / 2 * sizeof(array[0]),
                    &eventVector, &events[0]);

                // We need to get an event to wait for completion of the above
                // command on the command queue that provides the source buffer,
                // so it doesn't start modifying the buffer again too early

                // However, if we just do 'events[1] = events[0]', when we pass
                // the event to the next command, since it's associated with the
                // enqueueCopyBuffer command, it will cause an buffer sync.
                // (see OpenCL 1.2 spec Appendix A.1) that we don't want to happen

                // By doing this marker trick we ensure the event is not associated
                // to the enqueueCopyBuffer so it waits for completion, but doesn't sync.

                // (Or we could just do a cl::WaitForEvents and continue enqueuing
                // commands after the buffer copy is done to avoid those problems)
                std::vector<cl::Event> eventVector2 = {events[0]};
                devinfo[0].queue.enqueueMarkerWithWaitList(&eventVector2, &events[1]);
            }

            kernel.setArg(1, static_cast<cl_ulong>(array.size()));
            kernel.setArg(2, static_cast<cl_ulong>(phase));
            kernel.setArg(3, static_cast<cl_ulong>(subphase));

            if (NUM_DEVICES == 1 || needGatherScatter) {
                kernel.setArg(0, devinfo[0].buf);
                kernel.setArg(4, static_cast<cl_ulong>(0));
                std::vector<cl::Event> eventVector0 = {events[0]};
                devinfo[0].queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                    array.size(), cl::NullRange, &eventVector0, &events[0]);
            } else {
                kernel.setArg(0, devinfo[0].buf);
                kernel.setArg(4, static_cast<cl_ulong>(0));
                std::vector<cl::Event> eventVector0 = {events[0]};
                devinfo[0].queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                    array.size()/2, cl::NullRange, &eventVector0, &events[0]);

                kernel.setArg(0, devinfo[1].buf);
                kernel.setArg(4, static_cast<cl_ulong>(array.size()/2)); // NB: For correct swap direction
                std::vector<cl::Event> eventVector1 = {events[1]};
                devinfo[1].queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                    array.size()/2, cl::NullRange, &eventVector1, &events[1]);
            }

            if (needGatherScatter) {
                std::vector<cl::Event> eventVector = {events[0], events[1]};

                devinfo[1].queue.enqueueCopyBuffer(
                    devinfo[0].buf, devinfo[1].buf,
                    array.size() / 2 * sizeof(array[0]), 0,
                    array.size() / 2 * sizeof(array[0]),
                    &eventVector, &events[1]);

                // See similar code above for why this is done
                std::vector<cl::Event> eventVector2 = {events[1]};
                devinfo[1].queue.enqueueMarkerWithWaitList(&eventVector2, &events[0]);
            }
        }
    }

    // -------------------------
    // COPY DATA DEVICES -> HOST
    // -------------------------
    for (size_t d = 0; d < NUM_DEVICES; d++) {
        devinfo[d].queue.enqueueReadBuffer(devinfo[d].buf, CL_FALSE,
            0, array.size() / NUM_DEVICES * sizeof(array[0]),
            array.data() + d * array.size() / NUM_DEVICES,
            nullptr, &events[d]);
    }

    cl::Event::waitForEvents(events);

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ----------------
    // VALIDATE RESULTS
    // ----------------
    auto test_is_ok = array == expectedArray;
    std::cout << "Time:     " << duration_ms << " ms\n";
    std::cout << "Result:   " << (test_is_ok ? "OK" : "KO") << "\n";

    // TODOXXX: Without this code, the program will possibly hang after
    // after returning from main() when running on dOpenCL
    // This is because the host may still be waiting to be notified of the
    // completion of some of the events, but when the destructor for
    // _cl_platform_id just disconnects all connections to the compute nodes
    // without waiting for anything, so the completion notification
    // can never arrive and a deadlock happens
    // By finishing the command queue before returning, we make sure the host
    // has received all events completions locally so it's safe to disconnect
    //
    // Fix idea: Have the _cl_platform_id have a reference count that is
    // incremented and decremented by context creation/destruction
    // (similar to how e.g. programs reference their context).
    // On the destructor of _cl_platform_id wait until the reference count is
    // zero, which would mean all contexts have been released which would mean
    // all their associated resources have also been released, and so on recursively.
    //
    // Notes/clarifications:
    // * It is correct that releasing the command queues or events does not
    //   wait for their completion, see the docs. for clReleaseCommandQueue
    // * The previous cl::Event::waitForEvents runs on the compute node so
    //   even after it completes, maybe they host still hasn't received the
    //   event completion notification locally
    // * This can be reproduced very consistently and easily by just enqueuing
    //   a few commands and then returning from main without waiting for anything
    // * I think all/most other standalone_test's are affected, it's just that
    //   this one creates so many events that it's much more likely to happen
    for (auto &dev: devinfo)
        dev.queue.finish();

    return test_is_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
