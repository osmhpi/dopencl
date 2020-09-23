/******************************************************************************
 * This file is part of dOpenCL.
 * 
 * dOpenCL is an implementation of the OpenCL application programming
 * interface for distributed systems. See <http://dopencl.uni-muenster.de/>
 * for more information.
 * 
 * Developed by: Research Group Parallel and Distributed Systems
 *               Department of Mathematics and Computer Science
 *               University of Muenster, Germany
 *               <http://pvs.uni-muenster.de/>
 * 
 * Copyright (C) 2013  Philipp Kegel <philipp.kegel@uni-muenster.de>
 *
 * dOpenCL is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * dOpenCL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with dOpenCL. If not, see <http://www.gnu.org/licenses/>.
 * 
 * Permission to use dOpenCL for scientific, non-commercial work is
 * granted under the terms of the dOpenCL Academic License provided
 * appropriate credit is given. See the dOpenCL Academic License for
 * more details.
 * 
 * You should have received a copy of the dOpenCL Academic License
 * along with dOpenCL. If not, see <http://dopencl.uni-muenster.de/>.
 ******************************************************************************/

/*!
 * \file Buffer.cpp
 *
 * Buffer test suite (part of the memory test module)
 *
 * \date 2013-04-07
 * \author Philipp Kegel
 */

#include "utility.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <boost/test/unit_test.hpp>

#include <cstddef>
#include <vector>

namespace {

struct Context {
    cl_context context;
    cl_command_queue commandQueue;

    Context() {
        cl_platform_id platform = dcltest::getPlatform();
        cl_device_id device = dcltest::getDevice(platform);

        context = dcltest::createContext(1, &device);
        commandQueue = dcltest::createCommandQueue(context, device);

        BOOST_TEST_MESSAGE("Set up fixture");
    }

    ~Context() {
        // clean up
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);

        BOOST_TEST_MESSAGE("Teared down fixture");
    }
};

} // anonymous namespace

/* ****************************************************************************/

BOOST_FIXTURE_TEST_SUITE( Buffer, Context )

/* ****************************************************************************
 * Test cases
 ******************************************************************************/

BOOST_AUTO_TEST_CASE( CreateBuffer )
{
    const size_t SIZE = 1024;
    cl_int err = CL_SUCCESS;

    // create buffer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, SIZE, nullptr, &err);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    // clean up
    clReleaseMemObject(buffer);
}

BOOST_AUTO_TEST_CASE( CreateBufferCopyHostPtr )
{
    const size_t VEC_SIZE = 1024 * 1024;
    std::vector<cl_int> vec1(VEC_SIZE, 0), vec2(VEC_SIZE, 1), vec1copy;
    cl_int err = CL_SUCCESS;

    dcltest::fillVector(vec1, 1, 1); // initialize input data
    vec1copy = vec1;

    // create buffer from first host pointer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, VEC_SIZE * sizeof(cl_int), &vec1.front(), &err);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    // As a bonus test, make sure that CL_MEM_COPY_HOST_PTR did really copy our pointer,
    // by deallocating the associated memory now. Note that std::vector<T>.clear() does not guarantee this
    // See: https://web.archive.org/web/20190201121515/http://www.cplusplus.com/reference/vector/vector/clear/
    std::vector<cl_int>().swap(vec1);

    // download buffer to second host pointer
    err = clEnqueueReadBuffer(
            commandQueue,
            buffer,
            CL_TRUE,
            0, VEC_SIZE * sizeof(cl_int), &vec2.front(),
            0, nullptr, nullptr);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    BOOST_CHECK_MESSAGE(vec1copy == vec2, "Input and output buffers differ"); // compare input and output data

    // clean up
    clReleaseMemObject(buffer);
}

BOOST_AUTO_TEST_CASE( CreateBufferUseHostPtr )
{
    const size_t VEC_SIZE = 1024 * 1024;
    std::vector<cl_int> vec1(VEC_SIZE, 0), vec2(VEC_SIZE, 1);
    cl_int err = CL_SUCCESS;

    dcltest::fillVector(vec1, 1, 1); // initialize input data

    // create buffer from first host pointer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, VEC_SIZE * sizeof(cl_int), &vec1.front(), &err);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    // the vector data is referenced by the internal OpenCL buffer, so we can NOT destroy it here

    // download buffer to second host pointer
    err = clEnqueueReadBuffer(
            commandQueue,
            buffer,
            CL_TRUE,
            0, VEC_SIZE * sizeof(cl_int), &vec2.front(),
            0, nullptr, nullptr);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    BOOST_CHECK_MESSAGE(vec1 == vec2, "Input and output buffers differ"); // compare input and output data

    // clean up
    clReleaseMemObject(buffer);
}

// Tries to write and read back data from a window on a buffer with uneven offset and sizes,
// to try to trigger edge cases related to buffer alignment
BOOST_AUTO_TEST_CASE( ReadWriteBufferUnalignedWindow )
{
    const size_t VEC_SIZE = 1234567;
    std::vector<cl_uchar> vec1(VEC_SIZE, 0), vec2(VEC_SIZE, 1);

    cl_int err = CL_SUCCESS;

    dcltest::fillVector(vec1, static_cast<cl_uchar>(1), static_cast<cl_uchar>(1)); // initialize input data

    // create buffer
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, VEC_SIZE + 31 * 2, nullptr, &err);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    // write data to the buffer
    err = clEnqueueWriteBuffer(
            commandQueue, buffer, CL_TRUE,
            31, VEC_SIZE, &vec1.front(),
            0, nullptr, nullptr);

    // read data from buffer
    err = clEnqueueReadBuffer(
            commandQueue, buffer, CL_TRUE,
            31, VEC_SIZE, &vec2.front(),
            0, nullptr, nullptr);
    BOOST_REQUIRE_EQUAL(err, CL_SUCCESS);

    BOOST_CHECK_MESSAGE(vec1 == vec2, "Input and output buffers differ"); // compare input and output data

    // clean up
    clReleaseMemObject(buffer);
}

BOOST_AUTO_TEST_SUITE_END() // Buffer test suite
