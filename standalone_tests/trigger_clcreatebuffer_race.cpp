/* This will often trigger a race condition due in dOpenCL, which will almost
   surely cause a crash when I/O link compression is enabled. With it disabled,
   the program doesn't crash, but data gets silently mixed up between buffers.

   The cause is that while in dOpenCL data transfers should always be initiated
   by a request from a compute node, so they are effectively serialized,
   when calling clCreateBuffer with CL_MEM_USE_HOST_PTR, the host starts a
   data transfer unilaterally and outside the request loop. This can cause
   a pair of data transfer read-write commands to get executed in a
   different order in the host and the compute node, which causes the problem.
*/

#include <vector>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

int main(void)
{
    std::vector<unsigned char> datavec1(40000, 0);
    std::vector<unsigned char> datavec2(22222);
    for (size_t i = 0; i < datavec2.size(); i++) {
        datavec2[i] = (unsigned char)i * 7;
    }

    cl::Device device = cl::Device::getDefault();
    cl::Context context = cl::Context::getDefault();
    cl::CommandQueue queue(context, device);
    cl::Buffer buf(context, CL_MEM_READ_WRITE, datavec1.size());

    for (size_t i = 0; i < 100; i++) {
        // This will asynchronously initiate a data transfer through a request-response cycle
        queue.enqueueWriteBuffer(buf, false, 0, datavec1.size(), datavec1.data());
        // Try to mix up a data transfer before through clCreateBuffer(..., CL_MEM_USE_HOST_PTR, ...)
        cl::Buffer anotherbuf(context, CL_MEM_USE_HOST_PTR, datavec2.size(), datavec2.data());
    }


    return EXIT_SUCCESS;
}
