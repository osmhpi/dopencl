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
 * \file Context.cpp
 *
 * \date 2012-03-06
 * \author Philipp Kegel
 */

#include "Context.h"
#include "Device.h"

#include <dcl/CLEventCompletable.h>
#include <dcl/ComputeNode.h>
#include <dcl/ContextListener.h>
#include <dcl/DataTransfer.h>
#include <dcl/Device.h>
#include <dcl/Host.h>
#include <dcl/DCLTypes.h>

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION)
#include <lib842/stream/common.h>
#include <lib842/cl.h>
#endif

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace {

/*!
 * \brief Callback for context error
 */
static void onContextError(const char *errinfo, const void *private_info,
		size_t cb, void *user_data) {
	auto contextListener = static_cast<dcl::ContextListener *>(user_data);
	assert(contextListener != nullptr);
	contextListener->onError(errinfo, private_info, cb);
}

} /* unnamed namespace */

/* ****************************************************************************/

namespace dcld {

static cl::vector<cl::Device> convertToNativeDevicesList(const std::vector<dcl::Device *>& devices) {
    if (devices.empty()) { throw cl::Error(CL_INVALID_VALUE); }

    /* convert devices */
    cl::vector<cl::Device> nativeDevices;
    for (auto device : devices) {
        auto deviceImpl = dynamic_cast<Device *>(device);
        if (!deviceImpl) throw cl::Error(CL_INVALID_DEVICE);
        nativeDevices.push_back(deviceImpl->operator cl::Device());
    }
    return nativeDevices;
}

static cl::Context createNativeContext(
    const cl::vector<cl::Device> &nativeDevices,
    const cl::Platform& platform,
    const std::shared_ptr<dcl::ContextListener>& listener) {
    /* initialize context properties */
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform()),
        0 /* end of list */
    };

    return cl::Context(nativeDevices, properties, &onContextError, listener.get());
}

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
static lib842::CLDeviceDecompressor *createClDeviceCompressor(
    const cl::Context context,
    const cl::vector<cl::Device> &nativeDevices) {
    if (dcl::is_io_link_compression_enabled() && dcl::is_cl_io_link_compression_enabled()) {
        return new lib842::CLDeviceDecompressor(
                context, nativeDevices,
                lib842::stream::CHUNK_SIZE,
                lib842::stream::CHUNK_SIZE,
                dcl::is_cl_io_link_compression_mode_inline()
                    ? lib842::CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS
                    : lib842::CLDecompressorInputFormat::MAYBE_COMPRESSED_CHUNKS
        );
    }

    return nullptr;
}
#endif


Context::Context(
        dcl::Host& host,
        const std::vector<dcl::ComputeNode *>& computeNodes,
        const cl::Platform& platform,
        const std::vector<dcl::Device *>& devices,
        const std::shared_ptr<dcl::ContextListener>& listener) :
    Context(host, computeNodes, platform, convertToNativeDevicesList(devices), listener)
{}

Context::Context(
        dcl::Host& host,
        const std::vector<dcl::ComputeNode *>& computeNodes,
        const cl::Platform& platform,
        const cl::vector<cl::Device>& nativeDevices,
        const std::shared_ptr<dcl::ContextListener>& listener) :
    _host(host), _computeNodes(computeNodes), _listener(listener),
    _context(createNativeContext(nativeDevices, platform, listener)),
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    _cl842DeviceDecompressor(createClDeviceCompressor(_context, nativeDevices)),
#endif
    _ioCommandQueue(_context, nativeDevices.front()),
    _ioClOutDataTransferContext(_context, _ioCommandQueue)
{
    //	if (computeNodes.empty()) { throw cl::Error(CL_INVALID_VALUE); }
    /* TODO Remove self from list of compute nodes */
}

Context::~Context() { }

Context::operator cl::Context() const {
	return _context;
}

dcl::Host& Context::host() const {
	return _host;
}

const std::vector<dcl::ComputeNode *>& Context::computeNodes() const {
	return _computeNodes;
}

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
const lib842::CLDeviceDecompressor *Context::cl842DeviceDecompressor() const {
    return _cl842DeviceDecompressor.get();
}
#endif

const dcl::CLOutDataTransferContext& Context::ioClOutDataTransferContext() const {
    return _ioClOutDataTransferContext;
}

} /* namespace dcld */
