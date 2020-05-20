/* Count the number of matches of a certain pattern in text (like a regular expression) */
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#define MAX_RESULTS 24

static const std::string OPENCL_PROGRAM = R"V0G0N(
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

bool is_word_character(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           c == '_';
}

bool is_letter_character(char c) {
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z');
}

__kernel void count_matches(__global char *text, ulong size, __global ulong *results)
{
    ulong k = get_global_id(0);
    ulong match_start = k, match_end;

#if 1
    // Finds palindromic phrases (substrings) of up to a certain length
    // This is compute intensive due to the strategy used, and the amount
    // of compute can be regulated by the max. match length
    bool match;

    for (size_t i = 60; i >= 12; i--) {
        match = true;
        for (size_t j = 0; j < i/2; j++) {
            if (k+i-j-1 >= size ||                      // In bounds
                text[k+j] != text[k+i-j-1] ||           // Palindromic condition
                !is_letter_character(text[k+j]) ||      // Only ASCII words
                (j > 0 && text[k+j] == text[k+j-1]) ||  // No trivial matches like AAAA...AAAA
                (j > 1 && text[k+j] == text[k+j-2])) {  // No trivial matches like HAHA...HAHA
                match = false;
                break;
            }
        }

        if (match) {
             match_end = k+i;
             break;
        }
    }
#else
    // Matches like LC_ALL=C egrep -o "(t|T)he \w+( of the \w+){3,}",
    // except that egrep doesn't return overlapping matches
    // (this can be handled as a post-processing step if necessary)
    // Sample matches:
    // the development of the theory of the structure of the atomic 
    // the establishment of the Government of the Republic of the Marshall 
    // the squares of the differences of the roots of the original 
    // the basis of the report of the Governor of the concerned 
    // The History of the Formation of the Constitution of the United 
    // the unconstitutionality of the drain of the waters of the Punjab 
    // the study of the cause of the variation of the amount 
    // the squares of the differences of the roots of the original 

    if (k < size && (text[k] == 'T' || text[k] == 't'))   k++; else return;
    if (k < size && text[k] == 'h')                       k++; else return;
    if (k < size && text[k] == 'e')                       k++; else return;
    if (k < size && text[k] == ' ')                       k++; else return;
    if (k < size && is_word_character(text[k]))           k++; else return;
    while (k < size && is_word_character(text[k]))        k++;

    size_t count = 0;
    while (true) {
        if (k < size && text[k] == ' ')                       k++; else break;
        if (k < size && text[k] == 'o')                       k++; else break;
        if (k < size && text[k] == 'f')                       k++; else break;
        if (k < size && text[k] == ' ')                       k++; else break;
        if (k < size && text[k] == 't')                       k++; else break;
        if (k < size && text[k] == 'h')                       k++; else break;
        if (k < size && text[k] == 'e')                       k++; else break;
        if (k < size && text[k] == ' ')                       k++; else break;
        if (k < size && is_word_character(text[k]))           k++; else break;
        while (k < size && is_word_character(text[k]))        k++;
        match_end = k;
        count++;
    }

    bool match = count >= 3;
#endif

    if (match)
    {
        ulong my_counter = atom_inc(&results[0]);
        if (my_counter >= MAX_RESULTS) // Oops!
            return;
        results[1 + my_counter * 2] = match_start;
        results[1 + my_counter * 2 + 1] = match_end;
    }
}
)V0G0N";

typedef struct device_opencl
{
    size_t file_data_offset;
    size_t file_data_size;
    cl::CommandQueue queue;
    cl::Buffer buf;
    cl::Buffer results;
} device_opencl;

static std::vector<char> load_file_to_vector(const char *file_name) {
    std::ifstream file(file_name, std::ifstream::ate | std::ifstream::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    assert(file.tellg() == static_cast<std::streampos>(static_cast<size_t>(file.tellg())));
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    std::vector<char> file_data(size);
    file.read(reinterpret_cast<char *>(file_data.data()), file_data.size());
    return file_data;
}

static const std::uint64_t &ZERO = 0;

int main(int argc, char *argv[])
{
    if (argc != 2) {
        const char *program_name = argc > 0 ? argv[0] : "???";
        std::cout << "Usage: " << program_name << " file_name\n";
        return EXIT_FAILURE;
    }

    auto file_name = argv[1];
    auto file_data = load_file_to_vector(file_name);

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
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.size() < 1)
            throw std::runtime_error("No OpenCL devices available");
        std::cout << "(Using " << devices.size() << " devices)\n";

        cl::Context context(devices);
        cl::Program program(context, OPENCL_PROGRAM);
        std::ostringstream options;
        options << "-D MAX_RESULTS=" << MAX_RESULTS;
        program.build(devices, options.str().c_str());
        cl::Kernel kernel(program, "count_matches");

        std::vector<device_opencl> devinfo(devices.size());
        static constexpr size_t BLOCK_SIZE = 1048576;
        size_t size_blocks = (file_data.size() + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        for (cl_uint d = 0; d < devices.size(); d++) {
            device_opencl &dev = devinfo[d];
            dev.file_data_offset = d * size_blocks / devices.size();
            dev.file_data_size = std::min((d + 1) * size_blocks / devices.size(),
                                          file_data.size()) - dev.file_data_offset;
            dev.queue = cl::CommandQueue(context, devices[d]);
            dev.buf = cl::Buffer(context, CL_MEM_READ_ONLY, dev.file_data_size);
            dev.results = cl::Buffer(context, CL_MEM_READ_WRITE, (1 + 2 * MAX_RESULTS) * sizeof(std::uint64_t));
        }

        // --------------------
        // DATA WRITE & KERNELS
        // --------------------
        auto start_time = std::chrono::steady_clock::now();
        for (cl_uint d = 0; d < devices.size(); d++) {
            devinfo[d].queue.enqueueWriteBuffer(devinfo[d].buf, CL_FALSE, 0, devinfo[d].file_data_size,
                file_data.data() + devinfo[d].file_data_offset);

            devinfo[d].queue.enqueueWriteBuffer(devinfo[d].results, CL_FALSE, 0, sizeof(std::uint64_t), &ZERO);

            kernel.setArg(0, devinfo[d].buf);
            kernel.setArg(1, static_cast<cl_ulong>(devinfo[d].file_data_size));
            kernel.setArg(2, devinfo[d].results);
            devinfo[d].queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                (devinfo[d].file_data_size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE,
                cl::NullRange);
        }

        // --------------
        // GATHER RESULTS
        // --------------
        struct match { size_t start, end; };
        std::vector<match> matches;

        std::vector<std::uint64_t> match_buf(1 + 2 * MAX_RESULTS);
        for (cl_uint d = 0; d < devices.size(); d++) {
            devinfo[d].queue.enqueueReadBuffer(devinfo[d].results, CL_TRUE,
                0, (1 + 2 * MAX_RESULTS) * sizeof(std::uint64_t), match_buf.data());
            for (size_t i = 0; i < MAX_RESULTS && i < static_cast<size_t>(match_buf[0]); i++) {
                matches.push_back(match {
                    .start = static_cast<size_t>(match_buf[1 + 2 * i + 0] + devinfo[d].file_data_offset),
                    .end   = static_cast<size_t>(match_buf[1 + 2 * i + 1] + devinfo[d].file_data_offset),
                });
            }
        }

        // FIXME: If a match is cut between the data on two devices, we will not count it!
        //        This can be solved, but is tricky to get right, so we ignore it for now

        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        // ----------------
        // VALIDATE RESULTS
        // ----------------
        std::cout << "Matches: " << matches.size() << "\n";
        for (auto match : matches) {
            std::cout << "         \"";
            std::cout.write(&file_data[match.start], match.end - match.start);
            std::cout << "\"\n";
        }
        std::cout << "Time:    " << duration_ms << " ms\n" << std::flush;
    }

    return EXIT_SUCCESS;
}
