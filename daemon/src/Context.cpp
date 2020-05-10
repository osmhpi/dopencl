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

#include <lib842/stream/common.h>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
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
void onContextError(const char *errinfo, const void *private_info,
		size_t cb, void *user_data) {
	auto contextListener = static_cast<dcl::ContextListener *>(user_data);
	assert(contextListener != nullptr);
	contextListener->onError(errinfo, private_info, cb);
}

} /* unnamed namespace */

/* ****************************************************************************/

namespace dcld {

Context::Context(
        dcl::Host& host,
		const std::vector<dcl::ComputeNode *>& computeNodes,
		const cl::Platform& platform,
		const std::vector<dcl::Device *>& devices,
		const std::shared_ptr<dcl::ContextListener>& listener) :
    _host(host), _computeNodes(computeNodes), _listener(listener) {
    //	if (computeNodes.empty()) { throw cl::Error(CL_INVALID_VALUE); }
    if (devices.empty()) { throw cl::Error(CL_INVALID_VALUE); }

    /* TODO Remove self from list of compute nodes */

    /* TODO Use helper function for device conversion */
    /* convert devices */
    cl::vector<cl::Device> nativeDevices;
    for (auto device : devices) {
        auto deviceImpl = dynamic_cast<Device *>(device);
        if (!deviceImpl) throw cl::Error(CL_INVALID_DEVICE);
        nativeDevices.push_back(deviceImpl->operator cl::Device());
    }

    /* initialize context properties */
    cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM,
        reinterpret_cast<cl_context_properties>(platform()),
        0 /* end of list */
    };

	_context = cl::Context(nativeDevices, properties, &onContextError, _listener.get());
    _ioCommandQueue = cl::CommandQueue(_context, nativeDevices.front());

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    if (is_io_link_compression_enabled() && is_cl_io_link_compression_enabled()) {
        _cl842DeviceDecompressor = std::unique_ptr<lib842::CLDeviceDecompressor>(
            new lib842::CLDeviceDecompressor(
                _context, nativeDevices,
                lib842::stream::COMPR842_CHUNK_SIZE,
                lib842::stream::COMPR842_CHUNK_SIZE,
#if USE_CL_IO_LINK_COMPRESSION == 1 // Maybe compressed
                lib842::CLDecompressorInputFormat::MAYBE_COMPRESSED_CHUNKS
#else // Inplace compressed
                lib842::CLDecompressorInputFormat::INPLACE_COMPRESSED_CHUNKS
#endif
        ));
    }
#endif
}

Context::~Context() { }

Context::operator cl::Context() const {
	return _context;
}

dcl::Host& Context::host() const {
	return _host;
}

const cl::CommandQueue& Context::ioCommandQueue() const {
    return _ioCommandQueue;
}

const std::vector<dcl::ComputeNode *>& Context::computeNodes() const {
	return _computeNodes;
}

void Context::receiveBufferFromProcess(dcl::Process &process,
                                       const cl::CommandQueue &commandQueue,
                                       const cl::Buffer &buffer,
                                       dcl::transfer_id transferId,
                                       size_t offset,
                                       size_t size,
                                       const cl::vector<cl::Event> *eventWaitList,
                                       cl::Event *startEvent,
                                       cl::Event *endEvent) {
    return process.receiveDataToClBuffer(transferId, size, _context,
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
                                         _cl842DeviceDecompressor.get(),
#endif
                                         commandQueue, buffer, offset, eventWaitList, startEvent, endEvent);
}

void Context::sendBufferToProcess(dcl::Process &process,
                                  const cl::CommandQueue &commandQueue,
                                  const cl::Buffer &buffer,
                                  dcl::transfer_id transferId,
                                  size_t offset,
                                  size_t size,
                                  const cl::vector<cl::Event> *eventWaitList,
                                  cl::Event *startEvent,
                                  cl::Event *endEvent) {
    return process.sendDataFromClBuffer(transferId, size, _context,
                                        commandQueue, buffer, offset, eventWaitList, startEvent, endEvent);
}

} /* namespace dcld */
