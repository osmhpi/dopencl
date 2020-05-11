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
 * \file InputDataStream.cpp
 *
 * \date 2014-03-07
 * \author Philipp Kegel
 */

#include "InputDataStream.h"

#include "DataTransferImpl.h"
#include "DataTransferSentinelChecker.h"
#include "DataTransferProfiler.h"

#include <dcl/Completable.h>
#include <dcl/CLEventCompletable.h>
#include <dcl/DCLTypes.h>

#include <dcl/util/Logger.h>

#include <lib842/sw.h>
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
#include <lib842/hw.h>
#endif
#include <lib842/stream/decomp.h>

#include <boost/asio/buffer.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

namespace dclasio {

namespace comm {

#ifdef IO_LINK_COMPRESSION
static constexpr size_t NUM_CHUNKS_PER_NETWORK_BLOCK = lib842::stream::NUM_CHUNKS_PER_NETWORK_BLOCK;
static constexpr size_t CHUNK_SIZE = lib842::stream::COMPR842_CHUNK_SIZE;
static constexpr size_t COMPRESSIBLE_THRESHOLD = lib842::stream::COMPRESSIBLE_THRESHOLD;
static constexpr size_t NETWORK_BLOCK_SIZE = lib842::stream::NETWORK_BLOCK_SIZE;
#endif

InputDataStream::InputDataStream(boost::asio::ip::tcp::socket& socket)
    : _socket(socket), _read_state(receiving_state::idle) {
#ifdef IO_LINK_COMPRESSION
    if (dcl::is_io_link_compression_enabled()) {
        auto decompress842_func = optsw842_decompress;
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
        if (is_hw_io_link_compression_enabled())
            decompress842_func = hw842_decompress;
#endif
        auto num_threads = determine_io_link_compression_num_threads("DCL_IO_LINK_NUM_DECOMPRESS_THREADS");
        auto thread_policy = !determine_io_link_compression_spread_threads("DCL_IO_LINK_NUM_DECOMPRESS_SPREAD")
            ? lib842::stream::thread_policy::use_defaults
            : lib842::stream::thread_policy::spread_threads_among_numa_nodes;

        _decompress_thread_pool.reset(new lib842::stream::DataDecompressionStream(
            decompress842_func, num_threads, thread_policy,
            []() -> std::ostream& { return dcl::util::Logger << dcl::util::Error; },
            []() -> std::ostream& { return dcl::util::Logger << dcl::util::Debug; }
        ));
    }
#endif
}

std::shared_ptr<DataReceipt> InputDataStream::read(
        dcl::transfer_id transfer_id,
        size_t size, void *ptr, bool skip_compress_step,
        const std::shared_ptr<dcl::Completable> &trigger_event) {
    auto read(std::make_shared<DataReceipt>(transfer_id, size, ptr, skip_compress_step));
    if (trigger_event != nullptr) {
        // If a event to wait was given, start the read after the event callback
        trigger_event->setCallback([this, read](cl_int status) {
            if (status != CL_COMPLETE) {
                dcl::util::Logger << dcl::util::Error
                        << "InputDataStream: Wait for event (completable) for read enqueue failed"
                        << std::endl;
                read->onFinish(boost::system::errc::make_error_code(boost::system::errc::io_error),
                               0);
                return;
            }
            enqueue_read(read);
        });
    } else {
        // If no event to wait was given, start the read immediately
        enqueue_read(read);
    }
    return read;
}

void InputDataStream::enqueue_read(const std::shared_ptr<DataReceipt> &read) {
    std::vector<std::shared_ptr<DataReceipt>> reads = {read};

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    if (dcl::is_io_link_compression_enabled() && dcl::is_cl_io_link_compression_enabled() &&
        read->size() > CL_UPLOAD_BLOCK_SIZE) {
        // TODOXXX: The OpenCL-based decompression code does work in blocks of
        //          size CL_UPLOAD_BLOCK_SIZE, and for now, it splits its reads
        //          in blocks of this size. However, in order for all read/writes
        //          to match correctly, *all* transfers need to be split in blocks
        //          of this size. This is a huge hack and OpenCL-based decompression
        //          should eventually be integrated better so this isn't necessary
        reads.clear();

        size_t num_splits = (read->size() + CL_UPLOAD_BLOCK_SIZE - 1) / CL_UPLOAD_BLOCK_SIZE;
        dcl::transfer_id split_transfer_id = read->transferId();
        for (size_t i = 0; i < num_splits; i++, dcl::next_cl_split_transfer_id(split_transfer_id)) {
            size_t split_offset = i * CL_UPLOAD_BLOCK_SIZE;
            size_t split_size = std::min(read->size() - split_offset, CL_UPLOAD_BLOCK_SIZE);
            auto subread(std::make_shared<DataReceipt>(
                split_transfer_id, split_size,
                static_cast<uint8_t *>(read->ptr()) + split_offset,
                read->skip_compress_step()));
            reads.push_back(subread);
        }

        // Complete the main (non-split) transfer when the last split part finishes
        // TODOXXX error handling is not at all solid here
        reads.back()->setCallback([read](cl_int status) {
            auto ec = status == CL_SUCCESS
                 ? boost::system::error_code()
                 : boost::system::errc::make_error_code(boost::system::errc::io_error);
            read->onFinish(ec, read->size());
        });
    }
#endif

    std::unique_lock<std::mutex> lock(_readq_mtx);
    // If we were just waiting for this read to arrive to match a write,
    // don't even enqueue it, go straight to data receiving
    if (_read_state == receiving_state::waiting_for_read_matching_transfer_id &&
        _read_transfer_id == reads.front()->transferId()) {
        _read_state = receiving_state::receiving_data;
        for (auto it = ++reads.cbegin(); it != reads.cend(); it++)
            _readq.insert({ (*it)->transferId(), (*it) });
        lock.unlock();

        _read_op = reads.front();
        start_read();
        return;
    }

    // Otherwise, enqueue the request and wait for the matching write to arrive
    for (const auto &r : reads)
        _readq.insert({ r->transferId(), r });
    // If we are busy doing a receive for a transfer id or data, we are done,
    // the read queue will be processed out when the operation in course is done
    // But if we are idle, we need to start up the transfer ID matching process
    if (_read_state == receiving_state::idle) {
        _read_state = receiving_state::receiving_matching_transfer_id;
        lock.unlock();
        receive_matching_transfer_id();
    }
}

void InputDataStream::receive_matching_transfer_id() {
    boost_asio_async_read_with_sentinels(
        _socket, boost::asio::buffer(_read_transfer_id.data, _read_transfer_id.size()),
        [this](const boost::system::error_code& ec, size_t bytes_transferred) {
            std::unique_lock<std::mutex> lock(_readq_mtx);
            auto it = _readq.find(_read_transfer_id);
            if (it != _readq.end()) {
                // If the other end of the socket wants to do a send for a
                // receive we have enqueued, we can start receiving data
                auto readop = it->second;
                _readq.erase(it);
                _read_state = receiving_state::receiving_data;
                lock.unlock();

                _read_op = readop;
                start_read();
            } else {
                // Otherwise, we need to wait until that receive is enqueued
                _read_state = receiving_state::waiting_for_read_matching_transfer_id;
            }
        });
}

void InputDataStream::start_read() {
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(InputDataStream to " << _socket.remote_endpoint() << ") "
        << "Start read of size " << _read_op->size()
        << std::endl;
#endif
    _read_op->onStart();

#ifdef IO_LINK_COMPRESSION
    if (dcl::is_io_link_compression_enabled()) {
        _decompress_thread_pool->start();
        _read_io_total_bytes_transferred = 0;
        _read_io_num_blocks_remaining = _read_op->size() / NETWORK_BLOCK_SIZE;

        read_next_compressed_block();
        return;
    }
#endif

    boost_asio_async_read_with_sentinels(
                _socket, boost::asio::buffer(_read_op->ptr(), _read_op->size()),
                [this](const boost::system::error_code& ec, size_t bytes_transferred){
                        handle_read(ec, bytes_transferred); });
}

#ifdef IO_LINK_COMPRESSION
void InputDataStream::read_next_compressed_block() {
    if (_read_io_num_blocks_remaining > 0) {
        // Read header containing the offset and size of the chunks in the current block
        std::array<boost::asio::mutable_buffer, 2> buffers = {
                boost::asio::buffer(&_read_io_destination_offset, sizeof(size_t)),
                boost::asio::buffer(_read_io_buffer_sizes.data(), NUM_CHUNKS_PER_NETWORK_BLOCK * sizeof(size_t))
        };
        boost_asio_async_read_with_sentinels(_socket, buffers,
                [this] (const boost::system::error_code& ec, size_t bytes_transferred){
            _read_io_total_bytes_transferred += bytes_transferred;
            if (ec) {
                _decompress_thread_pool->finalize(true, [this, ec](bool) {
                    handle_read(ec, _read_io_total_bytes_transferred);
                });
                return;
            }

            assert(_read_io_destination_offset <= _read_op->size() - NETWORK_BLOCK_SIZE);
            for (auto read_io_buffer_size : _read_io_buffer_sizes) {
                assert(read_io_buffer_size > 0);
            }

            // Pre-allocate a temporary buffer to read compressed blocks
            size_t compressed_buffer_total_size = 0;
            for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && !_read_op->skip_compress_step()) {
                    compressed_buffer_total_size += _read_io_buffer_sizes[i];
                }
            }
            _read_io_compressed_buffer.reset(new uint8_t[compressed_buffer_total_size]);


            std::array<boost::asio::mutable_buffer, NUM_CHUNKS_PER_NETWORK_BLOCK> recv_buffers;
            for (size_t i = 0, compressed_buffer_offset = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && !_read_op->skip_compress_step()) {
                    // Read the compressed chunk into a secondary buffer to be decompressed later
                    recv_buffers[i] = boost::asio::buffer(
                        _read_io_compressed_buffer.get() + compressed_buffer_offset,
                        _read_io_buffer_sizes[i]);
                    compressed_buffer_offset += _read_io_buffer_sizes[i];
                } else {
                    // Read the chunk directly in its final destination in the destination buffer
                    uint8_t *destination = static_cast<uint8_t *>(_read_op->ptr()) + _read_io_destination_offset + i * CHUNK_SIZE;

                    if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && _read_op->skip_compress_step()) {
                        std::copy(LIB842_COMPRESSED_CHUNK_MARKER, LIB842_COMPRESSED_CHUNK_MARKER + sizeof(LIB842_COMPRESSED_CHUNK_MARKER), destination);
                        *reinterpret_cast<uint64_t *>((destination + sizeof(LIB842_COMPRESSED_CHUNK_MARKER))) = _read_io_buffer_sizes[i];
                        destination += CHUNK_SIZE - _read_io_buffer_sizes[i]; // Write compressed data at the end
                    } else {
                        assert(_read_io_buffer_sizes[i] == CHUNK_SIZE); // Chunk is read uncompressed
                    }

                    recv_buffers[i] = boost::asio::buffer(destination, _read_io_buffer_sizes[i]);
                }
            }

            boost_asio_async_read_with_sentinels(_socket, recv_buffers,
                [this] (const boost::system::error_code& ec, size_t bytes_transferred) {
                _read_io_total_bytes_transferred += bytes_transferred;
                if (ec) {
                    _decompress_thread_pool->finalize(true, [this, ec](bool) {
                        handle_read(ec, _read_io_total_bytes_transferred);
                    });
                    return;
                }

                // Push into the queue for decompression
                lib842::stream::DataDecompressionStream::decompress_block dm;
                dm.compress_buffer.reset(_read_io_compressed_buffer.release());
                bool should_uncompress_any = false;
                for (size_t i = 0, compressed_buffer_offset = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                    if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && !_read_op->skip_compress_step()) {
                        dm.chunks[i] = lib842::stream::DataDecompressionStream::decompress_chunk(
                            dm.compress_buffer.get() + compressed_buffer_offset,
                            _read_io_buffer_sizes[i],
                            static_cast<uint8_t *>(_read_op->ptr()) + _read_io_destination_offset + i * CHUNK_SIZE
                        );
                        compressed_buffer_offset += _read_io_buffer_sizes[i];
                        should_uncompress_any = true;
                    }
                }

                if (should_uncompress_any && !_decompress_thread_pool->push_block(std::move(dm))) {
                    handle_read(boost::system::errc::make_error_code(boost::system::errc::io_error),
                                _read_io_total_bytes_transferred);
                    return;
                }

                _read_io_num_blocks_remaining--;

                read_next_compressed_block();
            });
        });
    } else {
        // Always read the last incomplete block of the input uncompressed
        auto last_block_destination_ptr =  static_cast<uint8_t *>(_read_op->ptr()) + (_read_op->size() & ~(NETWORK_BLOCK_SIZE - 1));
        auto last_block_size = _read_op->size() & (NETWORK_BLOCK_SIZE - 1);

        boost_asio_async_read_with_sentinels(_socket, boost::asio::buffer(last_block_destination_ptr, last_block_size),
                                [this](const boost::system::error_code &ec, size_t bytes_transferred) {
            _read_io_total_bytes_transferred += bytes_transferred;
            if (ec) {
                _decompress_thread_pool->finalize(true, [this, ec](bool) {
                    handle_read(ec, _read_io_total_bytes_transferred);
                });
                return;
            }

            _decompress_thread_pool->finalize(false, [this](bool success) {
                auto ec = success ? boost::system::error_code()
                                  : boost::system::errc::make_error_code(boost::system::errc::io_error);
                handle_read(ec, _read_io_total_bytes_transferred);
            });
        });
    }
}
#endif

void InputDataStream::handle_read(
        const boost::system::error_code& ec,
        size_t bytes_transferred) {
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(InputDataStream to " << _socket.remote_endpoint() << ") "
        << "End read of size " << _read_op->size()
        << std::endl;
#endif
    _read_op->onFinish(ec, bytes_transferred);
    _read_op.reset();

    if (ec) {
        // TODO Handle errors
    }


    std::unique_lock<std::mutex> lock(_readq_mtx);
    if (!_readq.empty()) {
        // There are still receives enqueued, so try to match another one
        // now that we just finished one
        _read_state = receiving_state::receiving_matching_transfer_id;
        lock.unlock();
        receive_matching_transfer_id();
    } else {
        // All enqueues receives processed, we can go idle
        _read_state = receiving_state::idle;
    }
}


#if defined(CL_VERSION_1_2)
#define DOPENCL_MAP_WRITE_INVALIDATE_REGION CL_MAP_WRITE_INVALIDATE_REGION
#else
#define DOPENCL_MAP_WRITE_INVALIDATE_REGION CL_MAP_WRITE
#endif

void InputDataStream::readToClBuffer(
        dcl::transfer_id transferId,
        size_t size,
        const cl::Context &context,
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
        const lib842::CLDeviceDecompressor *cl842DeviceDecompressor,
#endif
        const cl::CommandQueue &commandQueue,
        const cl::Buffer &buffer,
        size_t offset,
        const cl::vector<cl::Event> *eventWaitList,
        cl::Event *startEvent,
        cl::Event *endEvent) {
    cl::Event myStartEvent, myEndEvent;

    if (startEvent == nullptr)
        startEvent = &myStartEvent;
    if (endEvent == nullptr)
        endEvent = &myEndEvent;

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    bool can_use_cl_io_link_compression = false;

    if (dcl::is_io_link_compression_enabled() && dcl::is_cl_io_link_compression_enabled() &&
        size > 0 && cl842DeviceDecompressor != nullptr) {
        if (offset != 0) {
            // TODOXXX It should be possible to handle nonzero offset cases here by passing this
            //         information to lib842, at least for 8-byte aligned cases
            dcl::util::Logger << dcl::util::Warning
                              << "Avoiding OpenCL hardware decompression due to non-zero buffer offset."
                              << std::endl;
        } else {
            can_use_cl_io_link_compression = true;
        }
    }

    if (can_use_cl_io_link_compression) {
#if USE_CL_IO_LINK_COMPRESSION == 1 // Maybe compressed
        static const size_t NUM_BUFFERS = 2;
        // TODOXXX: Allocate those somewhere more permanent, like on the command queue,
        //          so we don't need to allocate those buffers for every since transfer
        std::array<cl::Buffer, NUM_BUFFERS> workBuffers;
        for (auto &wb : workBuffers)
            wb = cl::Buffer(context, CL_MEM_READ_WRITE, CL_UPLOAD_BLOCK_SIZE);

        size_t num_splits = (size + CL_UPLOAD_BLOCK_SIZE - 1) / CL_UPLOAD_BLOCK_SIZE;

        std::vector<cl::Event> mapEvents(num_splits),
            unmapEvents(num_splits),
            decompressEvents(num_splits);
        std::vector<cl::UserEvent> receiveEvents;
        std::vector<void *> ptrs(num_splits);

        /* Setup */
        dcl::transfer_id split_transfer_id = transferId;
        for (size_t i = 0; i < num_splits + NUM_BUFFERS; i++) {
            const auto &wb = workBuffers[i % NUM_BUFFERS];

            if (i >= NUM_BUFFERS) {
                size_t split_offset = (i - NUM_BUFFERS) * CL_UPLOAD_BLOCK_SIZE;
                size_t split_size = std::min(size - split_offset, CL_UPLOAD_BLOCK_SIZE);

                /* Enqueue unmap buffer (implicit upload) */
                cl::vector<cl::Event> unmapWaitList = {receiveEvents[i - NUM_BUFFERS]};
                commandQueue.enqueueUnmapMemObject(wb, ptrs[i - NUM_BUFFERS], &unmapWaitList, &unmapEvents[i - NUM_BUFFERS]);
                // Rounds down (partial chunks are not compressed by InputDataStream)
                size_t chunksSize = split_size & ~(lib842::stream::COMPR842_CHUNK_SIZE - 1);
                if (chunksSize > 0) {
                    cl::vector<cl::Event> decompressWaitList = {unmapEvents[i - NUM_BUFFERS]};
                    cl842DeviceDecompressor->decompress(commandQueue,
                                                        wb, 0, chunksSize, cl::Buffer(nullptr),
                                                        buffer, split_offset, chunksSize, cl::Buffer(nullptr),
                                                        cl::Buffer(nullptr),
                                                        &decompressWaitList, &decompressEvents[i - NUM_BUFFERS]);
                } else {
                    decompressEvents[i - NUM_BUFFERS] = unmapEvents[i - NUM_BUFFERS];
                }
            }

            if (i < num_splits) {
                size_t split_offset = i * CL_UPLOAD_BLOCK_SIZE;
                size_t split_size = std::min(size - split_offset, CL_UPLOAD_BLOCK_SIZE);

                /* Enqueue map buffer */
                ptrs[i] = commandQueue.enqueueMapBuffer(
                    wb,
                    CL_FALSE,     // non-blocking map
                    DOPENCL_MAP_WRITE_INVALIDATE_REGION,
                    0, split_size,
                    eventWaitList, &mapEvents[i]);

                // schedule local data transfer
                cl::UserEvent receiveEvent(context);
                std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(mapEvents[i]));
                read(split_transfer_id, split_size, ptrs[i], true, mapDataCompletable)
                    ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));
                receiveEvents.push_back(receiveEvent);

                dcl::next_cl_split_transfer_id(split_transfer_id);
            }
        }

        *startEvent = mapEvents.front();
        *endEvent = decompressEvents.back();
#else // Inplace compressed
        size_t num_splits = (size + CL_UPLOAD_BLOCK_SIZE - 1) / CL_UPLOAD_BLOCK_SIZE;

        std::vector<cl::Event> mapEvents(num_splits),
            unmapEvents(num_splits),
            decompressEvents(num_splits);
        std::vector<void *> ptrs(num_splits);

        /* Enqueue map buffer */
        for (size_t i = 0; i < num_splits; i++) {
            size_t split_offset = i * CL_UPLOAD_BLOCK_SIZE;
            size_t split_size = std::min(size - split_offset, CL_UPLOAD_BLOCK_SIZE);
            ptrs[i] = commandQueue.enqueueMapBuffer(
                buffer,
                CL_FALSE,     // non-blocking map
                DOPENCL_MAP_WRITE_INVALIDATE_REGION,
                split_offset, split_size,
                eventWaitList, &mapEvents[i]);
        }

        dcl::transfer_id split_transfer_id = transferId;
        for (size_t i = 0; i < num_splits; i++, dcl::next_cl_split_transfer_id(split_transfer_id)) {
            size_t split_offset = i * CL_UPLOAD_BLOCK_SIZE;
            size_t split_size = std::min(size - i * CL_UPLOAD_BLOCK_SIZE, CL_UPLOAD_BLOCK_SIZE);

            // schedule local data transfer
            cl::UserEvent receiveEvent(context);
            std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(mapEvents[i]));
            read(split_transfer_id, split_size, ptrs[i], true, mapDataCompletable)
                ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));


            /* Enqueue unmap buffer (implicit upload) */
            cl::vector<cl::Event> unmapWaitList = {receiveEvent};
            commandQueue.enqueueUnmapMemObject(buffer, ptrs[i], &unmapWaitList, &unmapEvents[i]);
            // Rounds down (partial chunks are not compressed by InputDataStream)
            size_t chunksSize = split_size & ~(lib842::stream::COMPR842_CHUNK_SIZE - 1);
            if (chunksSize > 0) {
                cl::vector<cl::Event> decompressWaitList = {unmapEvents[i]};
                cl842DeviceDecompressor->decompress(commandQueue,
                                                    buffer, split_offset, chunksSize, cl::Buffer(nullptr),
                                                    buffer, split_offset, chunksSize, cl::Buffer(nullptr),
                                                    cl::Buffer(nullptr),
                                                    &decompressWaitList, &decompressEvents[i]);
            } else {
                decompressEvents[i] = unmapEvents[i];
            }
        }


        *startEvent = mapEvents.front();
        *endEvent = decompressEvents.back();
#endif
    } else {
#endif
        /* Enqueue map buffer */
        void *ptr = commandQueue.enqueueMapBuffer(
            buffer,
            CL_FALSE,     // non-blocking map
            DOPENCL_MAP_WRITE_INVALIDATE_REGION,
            offset, size,
            eventWaitList, startEvent);
        // schedule local data transfer
        cl::UserEvent receiveEvent(context);
        std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(*startEvent));
        read(transferId, size, ptr, false, mapDataCompletable)
            ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));
        /* Enqueue unmap buffer (implicit upload) */
        cl::vector<cl::Event> unmapWaitList = {receiveEvent};
        commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    }
#endif

    profile_transfer(profile_transfer_direction::receive, transferId, size, *startEvent, *endEvent);
}

} // namespace comm

} // namespace dclasio
