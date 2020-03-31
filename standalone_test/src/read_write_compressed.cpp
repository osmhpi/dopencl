/* Tests the download of buffers in a compressed state and their upload
 * from a compressed state.
 * This test requires IO_LINK_COMPRESSION in dOpenCL to be enabled!
 * It will not work on real hardware! */
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

typedef struct device_opencl
{
    size_t file_data_offset;
    size_t file_data_size;
    cl::CommandQueue queue;
    cl::Buffer buf;
} device_opencl;

#define NUM_TRANSFERS 15

static constexpr const size_t BUFFER_SIZE_SKIP_COMPRESS_STEP_BIT =
        (static_cast<size_t>(1) << (sizeof(size_t)*8-1));

static std::vector<uint8_t> load_file_to_vector(const char *file_name) {
    std::ifstream file(file_name, std::ifstream::ate | std::ifstream::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    assert(file.tellg() == static_cast<std::streampos>(static_cast<size_t>(file.tellg())));
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    std::vector<uint8_t> file_data(size);
    file.read(reinterpret_cast<char *>(file_data.data()), file_data.size());
    return file_data;
}

static void write_file_from_vector(const char *file_name, const std::vector<uint8_t> &file_data) {
    std::ofstream file(file_name);
    file.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    file.write(reinterpret_cast<const char *>(file_data.data()), file_data.size());
}

int main(int argc, char *argv[])
{
    if (argc != 4 || (strcmp(argv[1], "compress") != 0 &&
                      strcmp(argv[1], "decompress") != 0 &&
                      strcmp(argv[1], "send") != 0 &&
                      strcmp(argv[1], "receive") != 0)) {
        const char *program_name = argc > 0 ? argv[0] : "???";
        std::cout << "Usage: " << program_name
            << " compress|decompress|send|receive input output\n";
        return EXIT_FAILURE;
    }

    auto mode = argv[1];
    auto input_file_name = argv[2];
    auto output_file_name = argv[3];

    auto file_data = load_file_to_vector(input_file_name);

    int num_measurements = 1;
    const char *num_measurements_env = std::getenv("DCL_NUM_MEASUREMENTS");
    if (num_measurements_env != nullptr && std::atoi(num_measurements_env) > 0)
        num_measurements = std::atoi(num_measurements_env);

    for (int run = 0; run < num_measurements; run++) {
        // ---------------------
        // OPENCL INITIALIZATION
        // ---------------------
        cl::Platform platform;
        cl::Platform::get(&platform);

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.size() < 1)
            throw std::runtime_error("No OpenCL devices available");
        std::cout << "(Using " << devices.size() << " devices)\n";

        cl::Context context(devices);

        std::vector<device_opencl> devinfo(devices.size());
        static constexpr size_t BLOCK_SIZE = 1048576;
        size_t size_blocks = (file_data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        for (cl_uint d = 0; d < devices.size(); d++) {
            device_opencl &dev = devinfo[d];
            dev.file_data_offset = d * size_blocks / devices.size();
            dev.file_data_size = std::min((d + 1) * size_blocks / devices.size(),
                                          file_data.size()) - dev.file_data_offset;
            dev.queue = cl::CommandQueue(context, devices[d]);
            dev.buf = cl::Buffer(context, CL_MEM_READ_WRITE, dev.file_data_size);
        }

        // -----------------
        // DATA WRITE / READ
        // -----------------
        // "Warm up" the buffer (otherwise the first transfer takes abnormally long)
        std::vector<std::uint8_t> init(file_data.size(), 0);
        for (const auto &dev : devinfo) {
            dev.queue.enqueueWriteBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                init.data() + dev.file_data_offset);
        }
        for (const auto &dev : devinfo) { dev.queue.finish(); }


        std::chrono::time_point<std::chrono::steady_clock> start_time, end_time;
        if (strcmp(mode, "compress") == 0) {
            // Send data to device, not timed
            for (const auto &dev : devinfo) {
                dev.queue.enqueueWriteBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                    file_data.data() + dev.file_data_offset);
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }

            // Receive data from device, timed (Host transfers only, compute node compresses)
            start_time = std::chrono::steady_clock::now();
            for (size_t m = 0; m < NUM_TRANSFERS; m++) {
                for (const auto &dev : devinfo) {
                    dev.queue.enqueueReadBuffer(dev.buf, CL_FALSE, 0,
                        dev.file_data_size | BUFFER_SIZE_SKIP_COMPRESS_STEP_BIT,
                        file_data.data() + dev.file_data_offset);
                }
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }
            end_time = std::chrono::steady_clock::now();
        } else if (strcmp(mode, "decompress") == 0) {
            // Send data to device, timed (Host transfers only, compute node decompresses)
            start_time = std::chrono::steady_clock::now();
            for (size_t m = 0; m < NUM_TRANSFERS; m++) {
                for (const auto &dev : devinfo) {
                    dev.queue.enqueueWriteBuffer(dev.buf, CL_FALSE, 0,
                        dev.file_data_size | BUFFER_SIZE_SKIP_COMPRESS_STEP_BIT,
                        file_data.data() + dev.file_data_offset);
                }
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }
            end_time = std::chrono::steady_clock::now();

            // Receive data from device, not timed
            for (const auto &dev : devinfo) {
                dev.queue.enqueueReadBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                    file_data.data() + dev.file_data_offset);
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }
        } else if (strcmp(mode, "send") == 0) {
            // Send data to device, timed (host compresses, compute node decompresses)
            start_time = std::chrono::steady_clock::now();
            for (size_t m = 0; m < NUM_TRANSFERS; m++) {
                for (const auto &dev : devinfo) {
                    dev.queue.enqueueWriteBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                        file_data.data() + dev.file_data_offset);
                }
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }

            end_time = std::chrono::steady_clock::now();

            // Receive data from device, not timed
            for (const auto &dev : devinfo) {
                dev.queue.enqueueReadBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                    file_data.data() + dev.file_data_offset);
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }
        } else if (strcmp(mode, "receive") == 0) {
            // Send data to device, not timed
            for (const auto &dev : devinfo) {
                dev.queue.enqueueWriteBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                    file_data.data() + dev.file_data_offset);
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }

            // Receive data from device, timed (compute node compresses, host decompresses)
            start_time = std::chrono::steady_clock::now();
            for (size_t m = 0; m < NUM_TRANSFERS; m++) {
                for (const auto &dev : devinfo) {
                    dev.queue.enqueueReadBuffer(dev.buf, CL_FALSE, 0, dev.file_data_size,
                        file_data.data() + dev.file_data_offset);
                }
            }
            for (const auto &dev : devinfo) { dev.queue.finish(); }
            end_time = std::chrono::steady_clock::now();
        }
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / NUM_TRANSFERS;
        auto bandwidth_mbps = (static_cast<double>(file_data.size()) / (1024.0 * 1024.0)) / (static_cast<double>(duration_ms) / 1000.0);

        // ----------------
        // VALIDATE RESULTS
        // ----------------
        std::cout << "Time:      " << duration_ms << " ms (average over " << NUM_TRANSFERS << " transfers)\n";
        std::cout << "Bandwidth: " << bandwidth_mbps << " MB/s (average over " << NUM_TRANSFERS << " transfers)\n" << std::flush;
    }

    write_file_from_vector(output_file_name, file_data);

    return EXIT_SUCCESS;
}
