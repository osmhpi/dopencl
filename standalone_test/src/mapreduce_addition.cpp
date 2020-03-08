/* Tests doing a vector sum reduction on multiple devices (MapReduce-like pattern) */
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

static const std::string OPENCL_PROGRAM = R"V0G0N(
__kernel void reduce_sum( __global const uint *data, __global uint *partial_sums, __local uint *local_sums)
{
    size_t global_id = get_global_id(0), local_id = get_local_id(0), local_size = get_local_size(0);

    // Every group member loads one item of data
    local_sums[local_id] = data[global_id];

    for (size_t i = local_size / 2; i > 0; i >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        // Every iteration half of the threads wrt. the previous iteration
        // accumulate two values while the other half do nothing
        if (local_id < i) {
            local_sums[local_id] += local_sums[local_id + i];
        }
    }

    // Group leader stores final reduced sum of group to result buffer
    if (local_id == 0) {
        partial_sums[get_group_id(0)] = local_sums[0];
    }
}
)V0G0N";

typedef struct device_opencl
{
    cl::CommandQueue queue;
    cl::Buffer data;
    cl::Buffer partial_sums;
} device_opencl;

#ifndef BUF_SIZE
#define BUF_SIZE (1048576*10)
#endif
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 128
#endif

int main(void)
{
    std::vector<std::uint32_t> initial_buf(BUF_SIZE);
    for (size_t i = 0; i < BUF_SIZE; i++)
        initial_buf[i] = i;

    // ---------------------
    // OPENCL INITIALIZATION
    // ---------------------
    cl::Platform platform;
    cl::Platform::get(&platform);

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.size() < 1)
        throw std::runtime_error("No OpenCL devices available");
    #if 0 // For testing with a single device
    devices = std::vector<cl::Device>(4, devices[0]);
    #endif
    std::cout << "(Using " << devices.size() << " devices)\n";

    if ((BUF_SIZE % WORKGROUP_SIZE) != 0 || ((BUF_SIZE / WORKGROUP_SIZE) % devices.size()) != 0) {
        std::cout << "Invalid BUF_SIZE, WORKGROUP_SIZE and number of devices combination" << "\n";
        return EXIT_FAILURE;
    }
    const size_t BUF_SIZE_PER_DEVICE = BUF_SIZE / devices.size();
    const size_t PARTIAL_SUMS_PER_DEVICE = BUF_SIZE / (devices.size() * WORKGROUP_SIZE);

    cl::Context context(devices);
    cl::Program program(context, OPENCL_PROGRAM, true);
    cl::Kernel kernel(program, "reduce_sum");

    std::vector<device_opencl> devinfo(devices.size());
    for (cl_uint i = 0; i < devices.size(); i++) {
        device_opencl &dev = devinfo[i];
        dev.queue = cl::CommandQueue(context, devices[i]);
        dev.data = cl::Buffer(context, CL_MEM_READ_WRITE, BUF_SIZE_PER_DEVICE * sizeof(std::uint32_t));
        dev.partial_sums = cl::Buffer(context, CL_MEM_READ_WRITE, PARTIAL_SUMS_PER_DEVICE * sizeof(std::uint32_t));
    }

    // ------------------------
    // TRANSFER DATA TO DEVICES
    // ------------------------
    auto start_time = std::chrono::steady_clock::now();

    for (size_t i = 0; i < devices.size(); i++) {
        devinfo[i].queue.enqueueWriteBuffer(devinfo[i].data, CL_FALSE, 0,
                                            BUF_SIZE_PER_DEVICE * sizeof(std::uint32_t),
                                            initial_buf.data() + i * BUF_SIZE_PER_DEVICE);
    }

    // --------------
    // SUMS ON DEVICE
    // --------------
    for (size_t i = 0; i < devices.size(); i++) {
        kernel.setArg(0, devinfo[i].data);
        kernel.setArg(1, devinfo[i].partial_sums);
        kernel.setArg(2, WORKGROUP_SIZE * sizeof(std::uint32_t), nullptr);

        devinfo[i].queue.enqueueNDRangeKernel(kernel, cl::NullRange, BUF_SIZE / devices.size(), WORKGROUP_SIZE);
    }

    // ------------------------------
    // GATHER AND REDUCE PARTIAL SUMS
    // ------------------------------
    std::uint32_t actual_sum = 0;
    for (size_t i = 0; i < devices.size(); i++) {
        std::vector<std::uint32_t> partial_sums(PARTIAL_SUMS_PER_DEVICE);
        devinfo[i].queue.enqueueReadBuffer(devinfo[i].partial_sums, CL_TRUE, 0, partial_sums.size() * sizeof(std::uint32_t), partial_sums.data());
        for (auto v : partial_sums) {
            actual_sum += v;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ----------------
    // VALIDATE RESULTS
    // ----------------
    std::uint32_t expected_sum = 0;
    for (auto v : initial_buf)
        expected_sum += v;
    auto test_is_ok = expected_sum == actual_sum;

    std::cout << "Expected: " << expected_sum << "\n";
    std::cout << "Actual:   " << actual_sum << "\n";
    std::cout << "Time:     " << duration_ms << " ms\n";
    std::cout << "Result:   " << (test_is_ok ? "OK" : "KO") << "\n";
    return (test_is_ok) ? EXIT_SUCCESS : EXIT_FAILURE;
}
