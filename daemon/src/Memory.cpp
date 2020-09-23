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
 * \file Memory.cpp
 *
 * \date 2011-09-01
 * \author Philipp Kegel
 */

#include "Memory.h"

#include "Context.h"

#include <dcl/CLEventCompletable.h>
#include <dcl/DataTransfer.h>
#include <dcl/DCLException.h>
#include <dcl/DCLTypes.h>
#include <dcl/Process.h>

#include <dcl/util/Logger.h>

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#include <OpenCL/cl_wwu_dcl.h>
#else
#include <CL/cl2.hpp>
#include <CL/cl_wwu_dcl.h>
#endif

#include <cstddef>
#include <functional>
#include <memory>
#include <ostream>
#include <dclasio/message/RequestBufferTransfer.h>

/* ****************************************************************************/

namespace dcld {

/* ****************************************************************************
 * Memory object
 ******************************************************************************/

Memory::Memory(const std::shared_ptr<Context>& context) :
        _context(context) {
}

Memory::~Memory() { }

size_t Memory::size() const {
    return static_cast<cl::Memory>(*this).getInfo<CL_MEM_SIZE>();
}

bool Memory::isInput() const{
	return (static_cast<cl::Memory>(*this).getInfo<CL_MEM_FLAGS>() &
	        (CL_MEM_READ_ONLY | CL_MEM_READ_WRITE));
}

bool Memory::isOutput() const {
	return (static_cast<cl::Memory>(*this).getInfo<CL_MEM_FLAGS>() &
	        (CL_MEM_WRITE_ONLY | CL_MEM_READ_WRITE));
}

/* ****************************************************************************
 * Buffer
 ******************************************************************************/

Buffer::Buffer(
        const std::shared_ptr<Context>& context, cl_mem_flags flags,
		size_t size, dcl::object_id bufferId) :
	dcld::Memory(context)
{
    cl_mem_flags rwFlags = flags &
            (CL_MEM_READ_WRITE | CL_MEM_READ_ONLY | CL_MEM_WRITE_ONLY);
//    cl_mem_flags hostPtrFlags = flags &
//            (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR);
    cl_mem_flags allocHostPtr = flags &
            CL_MEM_ALLOC_HOST_PTR;

    /*
     * Always let the OpenCL implementation allocate the memory on the compute
     * node to ensure optimal performance.
     * If CL_MEM_USE_HOST_PTR is specified, the compute node should try to use
     * pinned memory to ensure optimal performance for frequent data transfers.
     */

    _buffer = cl::Buffer(*_context, rwFlags | allocHostPtr, size);

    cl_mem_flags hostPtrFlags = flags & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR);
    // If the buffer has been created with CL_MEM_USE_HOST_PTR or CL_MEM_COPY_HOST_PTR,
    // the data stays in the host until it is used for the first time it is used,
    // at which point it is synchronized in the compute node.
    _needsCreateBufferInitialSync = (hostPtrFlags != 0);
    _bufferId = bufferId;
}

Buffer::~Buffer() { }

Buffer::operator cl::Memory() const {
    return _buffer;
}

Buffer::operator cl::Buffer() const {
	return _buffer;
}

void Buffer::acquire(
        dcl::Process& process,
        const dcl::CLInDataTransferContext& clDataTransferContext,
        dcl::transfer_id transferId,
        cl::Event *releaseEvent,
        cl::Event *acquireEvent) {
    cl::Event mapEvent;

    dcl::util::Logger << dcl::util::Debug
            << "(SYN) Acquiring buffer from process '" << process.url() << '\''
            << std::endl;

    /* map buffer to host memory when releaseEvent is complete */
    cl::vector<cl::Event> receiveWaitList;
    if (releaseEvent != nullptr)
        receiveWaitList.push_back(*releaseEvent);

    /* enqueue data transfer to buffer */
    process.receiveDataToClBuffer(
            transferId, size(), clDataTransferContext, _buffer, 0,
            (receiveWaitList.empty() ? nullptr : &receiveWaitList), &mapEvent, acquireEvent);

    // Mark the buffer as needing no synchronization with the data
    // given to clCreateBuffer, since we have just acquired either
    // this data from the host,or a more recent version from a node
    _needsCreateBufferInitialSync = false;
}

void Buffer::release(
        dcl::Process& process,
        const dcl::CLOutDataTransferContext& clDataTransferContext,
        dcl::transfer_id transferId,
        const cl::Event& releaseEvent) const {
    cl::Event mapEvent, unmapEvent;

    dcl::util::Logger << dcl::util::Debug
            << "(SYN) Releasing buffer to process '" << process.url() << '\''
            << std::endl;

    /* enqueue data transfer from buffer when releaseEvent is complete */
    cl::vector<cl::Event> mapWaitList(1, releaseEvent);
    process.sendDataFromClBuffer(
            transferId, size(), clDataTransferContext, _buffer, 0,
            &mapWaitList, &mapEvent, &unmapEvent);
}

bool Buffer::checkCreateBufferInitialSync(dcl::Process&                       process,
                                          const dcl::CLInDataTransferContext& clDataTransferContext,
                                          cl::Event*                          acquireEvent) {
    if (!_needsCreateBufferInitialSync) {
        // Needs no synchronization or already synchronized
        return false;
    }

    // Request the host to transfer the buffer
    // TODOXXX: Is this the right place to do this? I think this transfer
    // and _bufferId should not belong here, but I can't find a better place
    dcl::transfer_id transferId = dcl::create_transfer_id();
    dclasio::message::RequestBufferTransfer msg(_bufferId, transferId);
    process.sendMessage(msg);

    // Download the buffer from the host
    acquire(process, clDataTransferContext, transferId, nullptr, acquireEvent);
    return true;
}

} /* namespace dcld */
