#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/opencl.h>

/*************
 * UTILITIES *
 *************/

// Macros for error checking
#define CHECK(x, msg) if (x) { \
                          fprintf(stderr, "%s at source %s:%d\n", msg, __FILE__, __LINE__); \
                          exit(EXIT_FAILURE); \
                      }
#define CHECK_OPENCL(err, msg) do { \
                                 cl_int myerr = err; \
                                 if (myerr != CL_SUCCESS) { \
                                    fprintf(stderr, "OpenCL error: %s (%d) at source %s:%d\n", \
                                            msg, myerr, __FILE__, __LINE__); \
                                    exit(EXIT_FAILURE); \
                                 } \
                             } while(0)

/******************
 * OPENCL PROGRAM *
 ******************/

static const char *OPENCL_PROGRAM =
"__kernel void increment_each(__global char *buf, ulong buf_size)\n"
"{\n"
"    size_t x = get_global_id(0);\n"
"\n"
"    //Make sure we do not go out of bounds\n"
"    if (x >= buf_size)\n"
"        return;\n"
"\n"
"    buf[x]++;"
"}\n"
;

typedef struct device_opencl
{
    cl_device_id device_id;
    cl_command_queue queue;
} device_opencl;

#define NUM_DEVICES 2
#define BUF_SIZE (1048576*30)

int main(void)
{
    // ---------------------
    // OPENCL INITIALIZATION
    // ---------------------
    cl_int err;

    cl_platform_id platform;
    CHECK_OPENCL(clGetPlatformIDs(1, &platform, NULL), "Could not get the OpenCL platform ID");

    cl_device_id device_ids[NUM_DEVICES];
    cl_uint available_devices;
    CHECK_OPENCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, NUM_DEVICES, device_ids, &available_devices),
                 "Could not get the number of available OpenCL GPU devices");
    CHECK(available_devices < NUM_DEVICES, "Not enough OpenCL devices available");

    cl_context context = clCreateContext(0, NUM_DEVICES, device_ids, NULL, NULL, &err);
    CHECK_OPENCL(err, "Could not create the OpenCL context");

    cl_program program = clCreateProgramWithSource(context, 1, &OPENCL_PROGRAM, NULL, &err);
    CHECK_OPENCL(err, "Could not create the OpenCL program");

    CHECK_OPENCL(clBuildProgram(program, NUM_DEVICES, device_ids, NULL, NULL, NULL),
                 "Could not build the OpenCL program");

    cl_kernel kernel = clCreateKernel(program, "increment_each", &err);
    CHECK_OPENCL(err, "Could not create the packed OpenCL kernel");

    cl_mem buf = clCreateBuffer(context, CL_MEM_READ_WRITE, BUF_SIZE, NULL, &err);
    CHECK_OPENCL(err, "Could not create the OpenCL 'from' buffer");

    device_opencl devices[NUM_DEVICES];
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl *dev = &devices[i];
        dev->device_id = device_ids[i];

#ifndef LEGACY_OPENCL_1_2
        dev->queue = clCreateCommandQueueWithProperties(context, dev->device_id, NULL, &err);
#else
        // This function is deprecated in OpenCL 2.0, but sadly NVIDIA only supports 1.2 currently
        dev->queue = clCreateCommandQueue(context, dev->device_id, 0, &err);
#endif
        CHECK_OPENCL(err, "Could not create the OpenCL queue");
    }

    // --------------
    // BUFFER FILLING
    // --------------
    // TODOXXX Why is this event necessary for dOpenCL and not on real HW (at least NVIDIA?)
    // Is this a bug in dOpenCL, or does NVIDIA use a stronger consistency model than that of the spec.?
    // See also: The tests/src/MemoryConsistency.cpp in the dOpenCL tree
    // See also: synchronize() method in daemon/src/CommandQueue.cpp on dOpenCL tree
    cl_event events[NUM_DEVICES+1];

    // NB: Avoiding clEnqueueFillBuffer since it's not supported by dOpenCL
    char *initial_buf = malloc(BUF_SIZE);
    CHECK(initial_buf == NULL, "Failed to allocate memory for the initial buffer");
    memset(initial_buf, 'A', BUF_SIZE);
    CHECK_OPENCL(clEnqueueWriteBuffer(devices[0].queue, buf, CL_TRUE, 0,
                                      BUF_SIZE, initial_buf, 0, NULL, &events[0]),
                 "Could not write the initial OpenCL buffer");
    free(initial_buf);

    // -------
    // KERNELS
    // -------
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl *dev = &devices[i];
        CHECK_OPENCL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf),
                     "Could not set the 1st kernel parameter");
        cl_ulong buf_size_ulong = BUF_SIZE;
        CHECK_OPENCL(clSetKernelArg(kernel, 1, sizeof(cl_ulong), &buf_size_ulong),
                     "Could not set the 2nd kernel parameter");
        // Execute the kernel over the entire range of the data set
        size_t block_size =  32;
        size_t global_size = (BUF_SIZE + block_size - 1) / block_size * block_size;
        clFinish(dev->queue);
        CHECK_OPENCL(clEnqueueNDRangeKernel(dev->queue, kernel, 1, NULL,
                                            &global_size, &block_size, 1, &events[i], &events[i+1]),
                     "Could not enqueue the kernel execution");
        clFinish(dev->queue);
    }

    // ---------------------
    // PRINT BUFFER (AFTER)
    // ---------------------
    printf("**AFTER KERNELS**\n");
    char buf_start[8], buf_end[8];
    CHECK_OPENCL(clEnqueueReadBuffer(devices[NUM_DEVICES-1].queue, buf, CL_TRUE, 0,
                                     sizeof(buf_start), buf_start, 0, NULL, NULL),
                 "Could not read the start of the OpenCL buffer");
    CHECK_OPENCL(clEnqueueReadBuffer(devices[NUM_DEVICES-1].queue, buf, CL_TRUE, BUF_SIZE - sizeof(buf_end),
                                     sizeof(buf_end), buf_end, 0, NULL, NULL),
                 "Could not read the start of the OpenCL buffer");
    printf("Buffer: %.*s...%.*s\n", (int)sizeof(buf_start), buf_start,
                                    (int)sizeof(buf_end), buf_end);
    memset(buf_start, 'A' + NUM_DEVICES, sizeof(buf_start));
    memset(buf_end, 'A' + NUM_DEVICES, sizeof(buf_end));
    printf("Expected: %.*s...%.*s\n", (int)sizeof(buf_start), buf_start,
                                      (int)sizeof(buf_end), buf_end);

    // -----------------------
    // OPENCL DEINITIALIZATION
    // -----------------------
    for (cl_uint i = 0; i < NUM_DEVICES; i++) {
        device_opencl *dev = &devices[i];
        clReleaseCommandQueue(dev->queue);
    }
    for (cl_uint i = 0; i < NUM_DEVICES+1; i++) {
        clReleaseEvent(events[i]);
    }
    clReleaseMemObject(buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);

    return EXIT_SUCCESS;
}
