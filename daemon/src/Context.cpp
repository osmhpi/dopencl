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
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#include <dcl/CLEventCompletable.h>
#include <dcl/DataTransfer.h>

#include <boost/log/trivial.hpp>

#ifdef USE_CL_IO_LINK_COMPRESSION
#error Not implemented yet (use USE_CL_IO_LINK_COMPRESSION_INPLACE instead).
#endif

#define PROFILE_SEND_RECEIVE_BUFFER

#ifdef PROFILE_SEND_RECEIVE_BUFFER
#include <chrono>
#include <atomic>

std::atomic<unsigned> profile_next_id;

struct profile_send_receive_buffer_times {
    unsigned id;
    size_t transfer_size;
    std::chrono::time_point<std::chrono::steady_clock> enqueue_time;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> end_time;
};
#endif

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
    // TODOXXX: This should not really be a shared_ptr,
    // but how should we initialize this at the beginning of the constructor instead?
    _cl842DeviceDecompressor = std::make_shared<CL842DeviceDecompressor>(
            _context, nativeDevices,
            dcl::DataTransfer::COMPR842_CHUNK_SIZE,
            dcl::DataTransfer::COMPR842_CHUNK_SIZE,
            CL842InputFormat::INPLACE_COMPRESSED_CHUNKS);
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
    cl::Event myStartEvent, myEndEvent;

    if (startEvent == nullptr)
        startEvent = &myStartEvent;
    if (endEvent == nullptr)
        endEvent = &myEndEvent;

    bool can_use_cl_io_link_compression = false;

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE)
    if (size > 0 && is_io_link_compression_enabled()) {
        if (offset != 0) {
            // TODOXXX It should be possible to handle nonzero offset cases here by passing this
            //         information to lib842, at least for 8-byte aligned cases
            BOOST_LOG_TRIVIAL(warning)
                                << "Avoiding OpenCL hardware decompression due to non-zero buffer offset."
                                << std::endl;
        } else {
            can_use_cl_io_link_compression = true;
        }
    }
#endif

    if (can_use_cl_io_link_compression) {
#ifdef IO_LINK_COMPRESSION
        static constexpr size_t SUPERBLOCK_MAX_SIZE = dcl::DataTransfer::SUPERBLOCK_MAX_SIZE;
        size_t num_superblocks = (size + SUPERBLOCK_MAX_SIZE - 1) / SUPERBLOCK_MAX_SIZE;

        std::vector<cl::Event> mapEvents(num_superblocks),
                               unmapEvents(num_superblocks),
                               decompressEvents(num_superblocks);
        std::vector<void *> ptrs(num_superblocks);

        /* Enqueue map buffer */
        for (size_t i = 0; i < num_superblocks; i++) {
            size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
            size_t superblock_size = std::min(size - superblock_offset, SUPERBLOCK_MAX_SIZE);
            ptrs[i] = commandQueue.enqueueMapBuffer(
                    buffer,
                    CL_FALSE,     // non-blocking map
#if defined(CL_VERSION_1_2)
                    CL_MAP_WRITE_INVALIDATE_REGION, // map for writing
#else
                    CL_MAP_WRITE, // map for writing
#endif
                    superblock_offset, superblock_size,
                    eventWaitList, &mapEvents[i]);
        }

        for (size_t i = 0; i < num_superblocks; i++) {
            size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
            size_t superblock_size = std::min(size - i * SUPERBLOCK_MAX_SIZE, SUPERBLOCK_MAX_SIZE);

            // schedule local data transfer
            cl::UserEvent receiveEvent(_context);
            std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(mapEvents[i]));
            process.receiveData(superblock_size, ptrs[i], true, mapDataCompletable)
                    ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));


            /* Enqueue unmap buffer (implicit upload) */
            VECTOR_CLASS<cl::Event> unmapWaitList = {receiveEvent};
            commandQueue.enqueueUnmapMemObject(buffer, ptrs[i], &unmapWaitList, &unmapEvents[i]);
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE)
            // Rounds down (partial chunks are not compressed by DataStream)
            size_t chunksSize = superblock_size & ~(dcl::DataTransfer::COMPR842_CHUNK_SIZE - 1);
            if (chunksSize > 0) {
                VECTOR_CLASS<cl::Event> decompressWaitList = {unmapEvents[i]};
                _cl842DeviceDecompressor->decompress(commandQueue,
                                                     buffer, superblock_offset, chunksSize, nullptr,
                                                     buffer, superblock_offset, chunksSize, nullptr,
                                                     nullptr,
                                                     &decompressWaitList, &decompressEvents[i]);
            } else {
                decompressEvents[i] = unmapEvents[i];
            }
#endif
        }

        
        *startEvent = mapEvents.front();
        *endEvent = decompressEvents.back();
#endif
    } else {
        /* Enqueue map buffer */
        void *ptr = commandQueue.enqueueMapBuffer(
                buffer,
                CL_FALSE,     // non-blocking map
#if defined(CL_VERSION_1_2)
            CL_MAP_WRITE_INVALIDATE_REGION, // map for writing
#else
            CL_MAP_WRITE, // map for writing
#endif
            offset, size,
            eventWaitList, startEvent);
        // schedule local data transfer
        cl::UserEvent receiveEvent(_context);
        std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(*startEvent));
        process.receiveData(size, ptr, false, mapDataCompletable)
                ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));
        /* Enqueue unmap buffer (implicit upload) */
        VECTOR_CLASS<cl::Event> unmapWaitList = {receiveEvent};
        commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);
    }

#ifdef PROFILE_SEND_RECEIVE_BUFFER
    auto profile_times = new profile_send_receive_buffer_times(); // TODOXXX: Release memory
    profile_times->id = profile_next_id++;
    profile_times->transfer_size = size;
    profile_times->enqueue_time = std::chrono::steady_clock::now();

    startEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
        auto profile_times = ((profile_send_receive_buffer_times *)user_data);
        profile_times->start_time = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(debug)
            << "(PROFILE) Receive with id " << profile_times->id << " of size " << profile_times->transfer_size
            << " started (ENQUEUE -> START) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    profile_times->start_time - profile_times->enqueue_time).count() << std::endl;
    }, profile_times);

    endEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
        auto profile_times = ((profile_send_receive_buffer_times *)user_data);
        profile_times->end_time = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(debug)
            << "(PROFILE) Receive with id " << profile_times->id << " of size " << profile_times->transfer_size
            << " decompressed (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    profile_times->end_time - profile_times->start_time).count() << std::endl;
    }, profile_times);
#endif
}

void Context::sendBufferToProcess(dcl::Process &process,
                                  const cl::CommandQueue &commandQueue,
                                  const cl::Buffer &buffer,
                                  size_t offset,
                                  size_t size,
                                  const VECTOR_CLASS<cl::Event> *eventWaitList,
                                  cl::Event *startEvent,
                                  cl::Event *endEvent) {
    cl::Event myStartEvent;
    if (startEvent == nullptr)
        startEvent = &myStartEvent;

    cl::UserEvent sendData(_context);

    /* Enqueue map buffer */
    void *ptr = commandQueue.enqueueMapBuffer(
            buffer,
            CL_FALSE,     // non-blocking map
            CL_MAP_READ, // map for reading
            offset, size,
            eventWaitList, startEvent);
    // schedule local data transfer
    std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(*startEvent));
    process.sendData(size, ptr, false, mapDataCompletable)
            ->setCallback(std::bind(&cl::UserEvent::setStatus, sendData, std::placeholders::_1));
    /* Enqueue unmap buffer (implicit upload) */
    VECTOR_CLASS<cl::Event> unmapWaitList = {sendData};
    commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);

#ifdef PROFILE_SEND_RECEIVE_BUFFER
    auto profile_times = new profile_send_receive_buffer_times(); // TODOXXX: Release memory
    profile_times->id = profile_next_id++;
    profile_times->transfer_size = size;
    profile_times->enqueue_time = std::chrono::steady_clock::now();

    startEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
        auto profile_times = ((profile_send_receive_buffer_times *)user_data);
        profile_times->start_time = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(debug)
            << "(PROFILE) Send with id " << profile_times->id << " of size " << profile_times->transfer_size
            << " started (ENQUEUE -> START) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    profile_times->start_time - profile_times->enqueue_time).count() << std::endl;
    }, profile_times);

    endEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
        auto profile_times = ((profile_send_receive_buffer_times *)user_data);
        profile_times->end_time = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(debug)
            << "(PROFILE) Send with id " << profile_times->id << " of size " << profile_times->transfer_size
            << " uploaded (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    profile_times->end_time - profile_times->start_time).count() << std::endl;
    }, profile_times);
#endif
}

} /* namespace dcld */
