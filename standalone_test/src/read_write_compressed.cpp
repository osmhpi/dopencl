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
#include <chrono>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

static constexpr const size_t BUFFER_SIZE_SKIP_COMPRESS_STEP_BIT =
        (static_cast<size_t>(1) << (sizeof(size_t)*8-1));

static std::vector<uint8_t> load_file_to_vector(const char *file_name) {
    std::ifstream file(file_name, std::ifstream::ate | std::ifstream::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    assert(file.tellg() < SIZE_MAX);
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
    if (argc != 4 || (strcmp(argv[1], "compress") != 0 && strcmp(argv[1], "decompress") != 0)) {
        const char *program_name = argc > 0 ? argv[0] : "???";
        std::cout
            << "Usage:\n"
            << "\t" << program_name << " compress input output\n"
            << "\t" << program_name << " decompress input output\n";
        return EXIT_FAILURE;
    }

    auto is_compress = strcmp(argv[1], "compress") == 0;
    auto input_file_name = argv[2];
    auto output_file_name = argv[3];

    auto file_data = load_file_to_vector(input_file_name);

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
    cl::Buffer buf(context, CL_MEM_READ_WRITE, file_data.size());

    // -----------------
    // DATA WRITE / READ
    // -----------------
    std::chrono::time_point<std::chrono::steady_clock> start_time, middle_time, end_time;
    start_time = std::chrono::steady_clock::now();
    queue.enqueueWriteBuffer(buf, CL_TRUE,
            0, file_data.size() | (!is_compress ? BUFFER_SIZE_SKIP_COMPRESS_STEP_BIT : 0),
            file_data.data());
    middle_time = std::chrono::steady_clock::now();
    queue.enqueueReadBuffer(buf, CL_TRUE,
            0, file_data.size() | (is_compress ? BUFFER_SIZE_SKIP_COMPRESS_STEP_BIT : 0),
            file_data.data());
    end_time = std::chrono::steady_clock::now();

    auto write_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(middle_time - start_time).count();
    auto read_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - middle_time).count();
    auto total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ----------------
    // VALIDATE RESULTS
    // ----------------
    write_file_from_vector(output_file_name, file_data);

    std::cout << "Time (write, relevant for 'decompress'): " << write_duration_ms << " ms\n";
    std::cout << "Time (read, relevant for 'compress'):    " << read_duration_ms << " ms\n";
    std::cout << "Time (total):                            " << total_duration_ms << " ms\n";

    return EXIT_SUCCESS;
}
