/* Tests downsizing a TIF image*/
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <chrono>
#include <memory>
#include <tiffio.h>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

typedef struct device_opencl
{
    std::uint32_t first_row;
    std::uint32_t num_rows;
    cl::CommandQueue queue;
    cl::Buffer raster_buf;
    cl::Buffer reduced_raster_buf;
} device_opencl;

#define REDUCTION_FACTOR 16

static const std::string OPENCL_PROGRAM = R"V0G0N(
__kernel void reduce_image( __global const uint *raster, __global uint *reduced_raster, uint width, uint height)
{
    uint reduced_width = (width + REDUCTION_FACTOR - 1) / REDUCTION_FACTOR;
    uint reduced_height = (height + REDUCTION_FACTOR - 1) / REDUCTION_FACTOR;

    size_t reduced_x = get_global_id(0), reduced_y = get_global_id(1);

    if (reduced_x >= reduced_width || reduced_y >= reduced_height)
        return;

    uint sums[4] = {0,0,0,0}, num_sums = 0;
    for (size_t y = reduced_y * REDUCTION_FACTOR; y < height && y < (reduced_y + 1) * REDUCTION_FACTOR; y++) {
        for (size_t x = reduced_x * REDUCTION_FACTOR; x < width && x < (reduced_x + 1) * REDUCTION_FACTOR; x++) {
            uint value = raster[y*width+x];
            sums[0] += (uchar)(value >> 24);
            sums[1] += (uchar)(value >> 16);
            sums[2] += (uchar)(value >> 8);
            sums[3] += (uchar)(value);
            num_sums++;
        }
    }

    uint avg_value = ((sums[0] / num_sums) << 24) |
                     ((sums[1] / num_sums) << 16) |
                     ((sums[2] / num_sums) << 8) |
                     ((sums[3] / num_sums));

    reduced_raster[reduced_y*reduced_width+reduced_x] = avg_value;
}
)V0G0N";

struct TIFFFileDeleter {
    void operator()(TIFF *tif) { TIFFClose(tif); }
};

static std::vector<std::uint32_t> read_tiff(const char *tiff_file_name,
    std::uint32_t &width, std::uint32_t &height)
{
    std::unique_ptr<TIFF, TIFFFileDeleter> tif(TIFFOpen(tiff_file_name, "r"));
    if (!tif)
        throw std::runtime_error("Can't open the input TIFF file");

    TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &width);
    TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &height);

    size_t npixels = width * height;
    std::vector<std::uint32_t> raster(npixels);
    if (!TIFFReadRGBAImage(tif.get(), width, height, raster.data(), 0))
        throw std::runtime_error("Can't read the input TIFF file");

    return raster;
}

static void write_tiff(const char *tiff_file_name,
    const std::vector<std::uint32_t> &raster, std::uint32_t width, std::uint32_t height)
{
    std::unique_ptr<TIFF, TIFFFileDeleter> tif(TIFFOpen(tiff_file_name, "w"));
    if (!tif)
        throw std::runtime_error("Can't open the output TIFF file");

    TIFFSetField(tif.get(), TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif.get(), TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif.get(), TIFFTAG_SAMPLESPERPIXEL, sizeof(std::uint32_t));
    TIFFSetField(tif.get(), TIFFTAG_BITSPERSAMPLE, 8);
    TIFFSetField(tif.get(), TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
    static const uint16 extras = EXTRASAMPLE_ASSOCALPHA;
    TIFFSetField(tif.get(), TIFFTAG_EXTRASAMPLES, 1, &extras);
    TIFFSetField(tif.get(), TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(tif.get(), width * sizeof(std::uint32_t)));

    std::vector<std::uint8_t> buf(TIFFScanlineSize(tif.get()));
    for (std::uint32_t row = 0; row < height; row++) {
        memcpy(buf.data(), &raster[(height-row-1)*width], sizeof(std::uint32_t) * width);
        if (TIFFWriteScanline(tif.get(), buf.data(), row, 0) < 0)
            throw std::runtime_error("Can't write the output TIFF file");
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        const char *program_name = argc > 0 ? argv[0] : "???";
        std::cout << "Usage: " << program_name << " input.tiff output.tiff\n";
        return EXIT_FAILURE;
    }

    std::uint32_t width, height;
    std::vector<std::uint32_t> raster = read_tiff(argv[1], width, height);
    std::cout
        << "TIFF WIDTH: " << width
        << ", HEIGHT: " << height
        << ", PIXELS: " << raster.size()
        << "\n";

    std::uint32_t reduced_width = (width + REDUCTION_FACTOR - 1) / REDUCTION_FACTOR;
    std::uint32_t reduced_height = (height + REDUCTION_FACTOR - 1) / REDUCTION_FACTOR;
    std::vector<std::uint32_t> reduced_raster(reduced_width * reduced_height);

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
    
    cl::Program program(context, OPENCL_PROGRAM);
    std::ostringstream options;
    options << "-D REDUCTION_FACTOR=" << REDUCTION_FACTOR;
    program.build(devices, options.str().c_str());
    cl::Kernel kernel(program, "reduce_image");

    std::vector<device_opencl> devinfo(devices.size());
    for (cl_uint d = 0; d < devices.size(); d++) {
        device_opencl &dev = devinfo[d];
        dev.first_row = (d * reduced_height / devices.size()) * REDUCTION_FACTOR;
        dev.num_rows = (d < (devices.size() - 1)
            ? ((d + 1) * reduced_height / devices.size()) * REDUCTION_FACTOR
            : height) - dev.first_row;
        dev.queue = cl::CommandQueue(context, devices[d]);
        dev.raster_buf = cl::Buffer(context, CL_MEM_READ_ONLY, raster.size() * sizeof(std::uint32_t));
        dev.reduced_raster_buf = cl::Buffer(context, CL_MEM_WRITE_ONLY, reduced_raster.size() * sizeof(std::uint32_t));
    }

    // ------------------------
    // TRANSFER DATA TO DEVICES
    // ------------------------
    auto start_time = std::chrono::steady_clock::now();

    for (cl_uint d = 0; d < devinfo.size(); d++) {
        devinfo[d].queue.enqueueWriteBuffer(devinfo[d].raster_buf, CL_FALSE, 0,
            devinfo[d].num_rows * width * sizeof(std::uint32_t),
            raster.data() + devinfo[d].first_row * width);
    }

    // --------------
    // SUMS ON DEVICE
    // --------------
    for (cl_uint d = 0; d < devinfo.size(); d++) {
        kernel.setArg(0, devinfo[d].raster_buf);
        kernel.setArg(1, devinfo[d].reduced_raster_buf);
        kernel.setArg(2, static_cast<cl_uint>(width));
        kernel.setArg(3, static_cast<cl_uint>(devinfo[d].num_rows));

        cl::NDRange localRange(16, 16);
        std::uint32_t reduced_num_rows = (devinfo[d].num_rows + REDUCTION_FACTOR - 1) / REDUCTION_FACTOR;
        cl::NDRange globalRange((reduced_width + localRange[0] - 1) & ~(localRange[0] - 1),
                                (reduced_num_rows + localRange[1] - 1) & ~(localRange[1] - 1));
        devinfo[d].queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalRange, localRange);
    }

    // ------------------------------
    // GATHER AND REDUCE PARTIAL SUMS
    // ------------------------------
    for (cl_uint d = 0; d < devinfo.size(); d++) {
        std::uint32_t reduced_first_row = devinfo[d].first_row / REDUCTION_FACTOR;
        std::uint32_t reduced_num_rows = (devinfo[d].num_rows + REDUCTION_FACTOR - 1) / REDUCTION_FACTOR;
        devinfo[d].queue.enqueueReadBuffer(devinfo[d].reduced_raster_buf, CL_TRUE, 0,
            reduced_num_rows * reduced_width * sizeof(std::uint32_t),
            reduced_raster.data() + reduced_first_row * reduced_width);
    }

    auto end_time = std::chrono::steady_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    // ----------------
    // VALIDATE RESULTS
    // ----------------
    write_tiff(argv[2], reduced_raster, reduced_width, reduced_height);

    std::cout << "Time:     " << duration_ms << " ms\n";
    return EXIT_SUCCESS;
}

