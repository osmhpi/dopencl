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

#include <dcl/Binary.h>
#include <dcl/ComputeNode.h>
#include <dcl/ContextListener.h>
#include <dcl/Device.h>
#include <dcl/Host.h>

#define __CL_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <cassert>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <dcl/CLEventCompletable.h>
#include <dcl/DataTransfer.h>

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
    VECTOR_CLASS<cl::Device> nativeDevices;
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

#ifdef IO_LINK_COMPRESSION
    _cl842DeviceDecompressor = std::make_shared<CL842DeviceDecompressor>(
            _context, nativeDevices, CL842_CHUNK_SIZE,false, true);
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
                                       size_t offset,
                                       size_t size,
                                       const VECTOR_CLASS<cl::Event> *eventWaitList,
                                       cl::Event *startEvent,
                                       cl::Event *endEvent) {
    cl::UserEvent copyData(_context);
    cl::Event unmapData;
#ifdef IO_LINK_COMPRESSION
    cl::Event decompressData;
    const bool skip_decompress_step = true;
#else
    const bool skip_decompress_step = false;
#endif

    /* Enqueue map buffer */
    void *ptr = commandQueue.enqueueMapBuffer(
            buffer,
            CL_FALSE,     // non-blocking map
            CL_MAP_WRITE, // map for writing
            offset, size,
            eventWaitList, startEvent);
    // schedule local data transfer
    std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(*startEvent));
    process.receiveData(size, ptr, skip_decompress_step, mapDataCompletable)
            ->setCallback(std::bind(&cl::UserEvent::setStatus, copyData, std::placeholders::_1));
    /* Enqueue unmap buffer (implicit upload) */
    VECTOR_CLASS<cl::Event> unmapWaitList = {copyData};
    commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, &unmapData);
#ifdef IO_LINK_COMPRESSION
    // decompress data
    // TODOXXX(U) handle offset parameter here! must add & pass to lib842
    size_t bufferSize = buffer.getInfo<CL_MEM_SIZE>();
    assert(offset == 0 && size == bufferSize);
    //printf("OFFSET: %zu SIZE: %zu TOTALSIZE: %zu\n", offset, size, buffer.getInfo<CL_MEM_SIZE>());
    VECTOR_CLASS<cl::Event> decompressWaitList = {unmapData};
    _cl842DeviceDecompressor->decompress(commandQueue, buffer, size, buffer, size,
                                                  &decompressWaitList, endEvent);
#else
    *endEvent = unmapData;
#endif
}

// TODOXXX(U) make symmetrical method for sending data
// TODOXXX(U) is this the right site?
// TODOXXX(U) add macro to enable/disable this behaviour?

} /* namespace dcld */
