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

#ifdef IO_LINK_COMPRESSION
#include <lib842/sw.h>
#include <lib842/stream/decomp.h>
#endif
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
#include <lib842/hw.h>
#endif
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION)
#include <lib842/cl.h>
#endif

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
#include <stdlib.h> // Hacky use of C11 aligned_alloc,
                    // since std::aligned_alloc is not available until C++17

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

namespace dclasio {

namespace comm {

#ifdef IO_LINK_COMPRESSION
using lib842::stream::CHUNK_SIZE;
using lib842::stream::NUM_CHUNKS_PER_BLOCK;
using lib842::stream::BLOCK_SIZE;
using lib842::stream::COMPRESSIBLE_THRESHOLD;
#endif

InputDataStream::InputDataStream(boost::asio::ip::tcp::socket& socket)
    : _socket(socket), _read_state(receiving_state::idle) {
#ifdef IO_LINK_COMPRESSION
    if (dcl::is_io_link_compression_enabled()) {
        _impl842 = get_optsw842_implementation();
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
        if (is_hw_io_link_compression_enabled()) {
            _impl842 = get_hw842_implementation();
        }
#endif
        auto num_threads = determine_io_link_compression_num_threads("DCL_IO_LINK_NUM_DECOMPRESS_THREADS");
        auto thread_policy = !determine_io_link_compression_spread_threads("DCL_IO_LINK_SPREAD_THREADS")
            ? lib842::stream::thread_policy::use_defaults
            : lib842::stream::thread_policy::spread_threads_among_numa_nodes;

        _decompress_thread_pool.reset(new lib842::stream::DataDecompressionStream(
            *_impl842, num_threads, thread_policy,
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
    return read(transfer_id, size, ptr, skip_compress_step, dcl::transfer_id(), 0, trigger_event);
}

std::shared_ptr<DataReceipt> InputDataStream::read(
        dcl::transfer_id transfer_id,
        size_t size, void *ptr, bool skip_compress_step,
        dcl::transfer_id split_transfer_next_id, size_t split_transfer_global_offset,
        const std::shared_ptr<dcl::Completable> &trigger_event) {
    auto read(std::make_shared<DataReceipt>(transfer_id, size, ptr, skip_compress_step,
        split_transfer_next_id, split_transfer_global_offset));
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
    std::unique_lock<std::mutex> lock(_readq_mtx);
    // If we were just waiting for this read to arrive to match a write,
    // don't even enqueue it, go straight to data receiving
    if (_read_state == receiving_state::waiting_for_read_matching_transfer_id &&
        _read_transfer_id == read->transferId()) {
        _read_state = receiving_state::receiving_data;
        lock.unlock();

        _read_op = read;
        start_read();
        return;
    }

    // Otherwise, enqueue the request and wait for the matching write to arrive
    _readq.insert({ read->transferId(), read });
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
            assert(!ec); // TODO: How to handle an error here?
            on_transfer_id_received();
        });
}

void InputDataStream::on_transfer_id_received() {
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
    if (dcl::is_io_link_compression_enabled() && _read_op->size() >= BLOCK_SIZE) {
        _read_io_total_bytes_transferred = 0;
        _read_io_num_blocks_remaining = _read_op->size() / BLOCK_SIZE;

        if (!_read_op->skip_compress_step()) {
            _decompress_thread_pool->start(_read_op->ptr());
            read_next_compressed_block_and_decompress();
        } else {
            read_next_compressed_block_skip_compression_step();
        }
        return;
    }
#endif

    boost_asio_async_read_with_sentinels(
                _socket, boost::asio::buffer(_read_op->ptr(), _read_op->size()),
                [this](const boost::system::error_code& ec, size_t bytes_transferred){
                        handle_read(ec, bytes_transferred); });
}

#ifdef IO_LINK_COMPRESSION
void InputDataStream::read_next_compressed_block_and_decompress() {
    if (_read_io_num_blocks_remaining > 0) {
        // Read header containing the offset and size of the chunks in the current block
        std::array<boost::asio::mutable_buffer, 2> buffers = {
                boost::asio::buffer(&_read_io_block_offset, sizeof(size_t)),
                boost::asio::buffer(_read_io_block_sizes.data(), NUM_CHUNKS_PER_BLOCK * sizeof(size_t))
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

            // If this is a split transfer, adjust the offset according to the part
            // of the transfer we are currently receiving
            assert(_read_io_block_offset >= _read_op->split_transfer_global_offset());
            _read_io_block_offset -= _read_op->split_transfer_global_offset();

            // Validate received values
            assert(_read_io_block_offset <= _read_op->size() - BLOCK_SIZE);
            for (auto read_io_buffer_size : _read_io_block_sizes) {
                assert(read_io_buffer_size > 0 &&
                       (read_io_buffer_size <= COMPRESSIBLE_THRESHOLD ||
                        read_io_buffer_size == CHUNK_SIZE));
            }

            std::array<boost::asio::mutable_buffer, NUM_CHUNKS_PER_BLOCK> recv_buffers;
            bool should_uncompress_any = false;
            for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
                if (_read_io_block_sizes[i] == CHUNK_SIZE) {
                    // Read the chunk directly in its final destination in the destination buffer
                    auto destination = static_cast<uint8_t *>(_read_op->ptr()) + _read_io_block_offset + i * CHUNK_SIZE;
                    recv_buffers[i] = boost::asio::buffer(destination, _read_io_block_sizes[i]);

                    // Null out the chunk in the block so the decompression process will ignore it
                    _read_io_block_sizes[i] = 0;
                } else {
                    should_uncompress_any = true;
                }
            }

            if (should_uncompress_any) {
                _read_io_block_opt.emplace();
                auto &_read_io_block = _read_io_block_opt.value();
                _read_io_block.offset = _read_io_block_offset;
                _read_io_block.sizes = _read_io_block_sizes;
                _read_io_block.chunk_padding = CHUNK_SIZE;
                _read_io_block.source = nullptr;

                // NB: Note that space in chunk_buffer is wasted if there are
                // uncompressible chunks. This makes the logic much simpler though,
                // so don't think 'condensing' the chunks just to save a few bytes
                // of memory is a good idea...
                uint8_t *chunk_buffer = _read_io_block.allocate_buffer(
                    _impl842->required_alignment, CHUNK_SIZE);

                for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
                    if (_read_io_block_sizes[i] > 0) {
                        // Read the compressed chunk into a secondary buffer to be decompressed later
                        auto destination = chunk_buffer + i * CHUNK_SIZE;
                        recv_buffers[i] = boost::asio::buffer(destination, _read_io_block.sizes[i]);
                    }
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
                if (_read_io_block_opt && !_decompress_thread_pool->push_block(std::move(_read_io_block_opt.value()))) {
                    _decompress_thread_pool->finalize(true, [this](bool) {
                        handle_read(boost::system::errc::make_error_code(boost::system::errc::io_error),
                                    _read_io_total_bytes_transferred);
                    });
                    return;
                }
                _read_io_block_opt = boost::none;

                _read_io_num_blocks_remaining--;

                read_next_compressed_block_and_decompress();
            });
        });
    } else {
        // Always read the last incomplete block of the input uncompressed
        auto last_block_destination_ptr =  static_cast<uint8_t *>(_read_op->ptr()) + (_read_op->size() & ~(BLOCK_SIZE - 1));
        auto last_block_size = _read_op->size() & (BLOCK_SIZE - 1);

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

void InputDataStream::read_next_compressed_block_skip_compression_step() {
    if (_read_io_num_blocks_remaining > 0) {
        // Read header containing the offset and size of the chunks in the current block
        std::array<boost::asio::mutable_buffer, 2> buffers = {
                boost::asio::buffer(&_read_io_block_offset, sizeof(size_t)),
                boost::asio::buffer(_read_io_block_sizes.data(), NUM_CHUNKS_PER_BLOCK * sizeof(size_t))
        };
        boost_asio_async_read_with_sentinels(_socket, buffers,
                [this] (const boost::system::error_code& ec, size_t bytes_transferred){
            _read_io_total_bytes_transferred += bytes_transferred;
            if (ec) {
                handle_read(ec, _read_io_total_bytes_transferred);
                return;
            }

            // If this is a split transfer, adjust the offset according to the part
            // of the transfer we are currently receiving
            assert(_read_io_block_offset >= _read_op->split_transfer_global_offset());
            _read_io_block_offset -= _read_op->split_transfer_global_offset();

            // Validate received values
            assert(_read_io_block_offset <= _read_op->size() - BLOCK_SIZE);
            for (auto read_io_buffer_size : _read_io_block_sizes) {
                assert(read_io_buffer_size > 0);
            }

            std::array<boost::asio::mutable_buffer, NUM_CHUNKS_PER_BLOCK> recv_buffers;
            for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
                // Read the chunk directly in its final destination in the destination buffer
                auto destination = static_cast<uint8_t *>(_read_op->ptr()) + _read_io_block_offset + i * CHUNK_SIZE;

                if (_read_io_block_sizes[i] <= COMPRESSIBLE_THRESHOLD) {
                    std::copy(LIB842_COMPRESSED_CHUNK_MARKER, LIB842_COMPRESSED_CHUNK_MARKER + sizeof(LIB842_COMPRESSED_CHUNK_MARKER), destination);
                    *reinterpret_cast<uint64_t *>((destination + sizeof(LIB842_COMPRESSED_CHUNK_MARKER))) = _read_io_block_sizes[i];
                    destination += CHUNK_SIZE - _read_io_block_sizes[i]; // Write compressed data at the end
                } else {
                    assert(_read_io_block_sizes[i] == CHUNK_SIZE); // Chunk is read uncompressed
                }

                recv_buffers[i] = boost::asio::buffer(destination, _read_io_block_sizes[i]);
            }

            boost_asio_async_read_with_sentinels(_socket, recv_buffers,
                [this] (const boost::system::error_code& ec, size_t bytes_transferred) {
                _read_io_total_bytes_transferred += bytes_transferred;
                if (ec) {
                    handle_read(ec, _read_io_total_bytes_transferred);
                    return;
                }

                _read_io_num_blocks_remaining--;

                read_next_compressed_block_skip_compression_step();
            });
        });
    } else {
        // Always read the last incomplete block of the input uncompressed
        auto last_block_destination_ptr =  static_cast<uint8_t *>(_read_op->ptr()) + (_read_op->size() & ~(BLOCK_SIZE - 1));
        auto last_block_size = _read_op->size() & (BLOCK_SIZE - 1);

        boost_asio_async_read_with_sentinels(_socket, boost::asio::buffer(last_block_destination_ptr, last_block_size),
                                [this](const boost::system::error_code &ec, size_t bytes_transferred) {
            _read_io_total_bytes_transferred += bytes_transferred;
            handle_read(ec, _read_io_total_bytes_transferred);
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
    auto split_transfer_next_id = _read_op->split_transfer_next_id();
    _read_op->onFinish(ec, bytes_transferred);
    _read_op.reset();

    if (ec) {
        // TODO Handle errors
    }

    // Skip transfer ID write for between split transfers
    if (!split_transfer_next_id.is_nil()) {
        _read_transfer_id = split_transfer_next_id;
        on_transfer_id_received();
        return;
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
        const dcl::CLInDataTransferContext &clDataTransferContext,
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
    auto can_use_cl_io_link_compression =
        dcl::is_io_link_compression_enabled() && dcl::is_cl_io_link_compression_enabled() &&
        size > 0 && clDataTransferContext.cl842DeviceDecompressor() != nullptr;

    if (can_use_cl_io_link_compression && (offset % 8) != 0) {
        // The OpenCL 842 decompressor requires the offset to be 8-byte aligned,
        // since it type-puns the buffer to a uint64_t
        dcl::util::Logger << dcl::util::Warning
                          << "Avoiding OpenCL hardware decompression due to non 8 byte buffer offset."
                          << std::endl;
        can_use_cl_io_link_compression = false;
    }

    if (can_use_cl_io_link_compression) {
        if (dcl::is_cl_io_link_compression_mode_inplace()) {
            readToClBufferWithClInplaceDecompression(
                transferId, size, clDataTransferContext,
                buffer, offset, eventWaitList, startEvent, endEvent);
        } else {
            readToClBufferWithClTemporaryDecompression(
                transferId, size, clDataTransferContext,
                buffer, offset, eventWaitList, startEvent, endEvent);
        }
        profile_transfer(profile_transfer_direction::receive, transferId, size,
                         *startEvent, *endEvent);
        return;
    }
#endif

    readToClBufferWithNonClDecompression(
        transferId, size, clDataTransferContext.context(), clDataTransferContext.commandQueue(),
        buffer, offset, eventWaitList, startEvent, endEvent);
    profile_transfer(profile_transfer_direction::receive, transferId, size,
                     *startEvent, *endEvent);
}


void InputDataStream::readToClBufferWithNonClDecompression(
        dcl::transfer_id transferId,
        size_t size,
        const cl::Context &context,
        const cl::CommandQueue &commandQueue,
        const cl::Buffer &buffer,
        size_t offset,
        const cl::vector<cl::Event> *eventWaitList,
        cl::Event *startEvent,
        cl::Event *endEvent) {
    /* Enqueue map buffer */
    void *ptr = commandQueue.enqueueMapBuffer(
        buffer,
        CL_FALSE,     // non-blocking map
        DOPENCL_MAP_WRITE_INVALIDATE_REGION,
        offset, size,
        eventWaitList, startEvent);
    // schedule local data transfer
    cl::UserEvent receiveEvent(context);
    auto mapDataCompletable(std::make_shared<dcl::CLEventCompletable>(*startEvent));
    read(transferId, size, ptr, false, mapDataCompletable)
        ->setCallback([receiveEvent](cl_int status) mutable { receiveEvent.setStatus(status); });
    /* Enqueue unmap buffer (implicit upload) */
    cl::vector<cl::Event> unmapWaitList = {receiveEvent};
    commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);
}

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
static std::vector<dcl::transfer_id> generateClSplitTransferIds(dcl::transfer_id transferId, size_t numSplits) {
    std::vector<dcl::transfer_id> splitTransferIds(numSplits + 1);
    // The first part needs to have the original transfer ID, in order to match the write
    splitTransferIds[0] = transferId;
    // The intermediate positions are generated (they only matter internally to link the transfer)
    for (size_t i = 1; i < numSplits; i++)
        splitTransferIds[i] = dcl::create_transfer_id();
    // The last position (index [numSplits]) stays at the default value (=nil),
    // which will represent that the split transfer ends
    return splitTransferIds;
}

void InputDataStream::readToClBufferWithClTemporaryDecompression(
        dcl::transfer_id transferId,
        size_t size,
        const dcl::CLInDataTransferContext &clDataTransferContext,
        const cl::Buffer &buffer,
        size_t offset,
        const cl::vector<cl::Event> *eventWaitList,
        cl::Event *startEvent,
        cl::Event *endEvent) {
    auto commandQueue = clDataTransferContext.commandQueue();

    // Avoid re-allocating new temporary buffers for each transfer by storing
    // them in the CL data transfer context. This is safe because each context
    // is associated with a command queue, and so far out of order queues are
    // not supported, so only a single transfer is using the buffers at a time
    auto &workBuffers = clDataTransferContext.cl842WorkBuffers();
    static const size_t NUM_BUFFERS = workBuffers.size();
    for (auto &wb : workBuffers) {
        if (wb.get() != nullptr)
            break;
        wb = cl::Buffer(clDataTransferContext.context(), CL_MEM_READ_ONLY, CL_UPLOAD_BLOCK_SIZE);
    }

    size_t num_splits = (size + CL_UPLOAD_BLOCK_SIZE - 1) / CL_UPLOAD_BLOCK_SIZE;
    auto splitTransferIds = generateClSplitTransferIds(transferId, num_splits);

    std::vector<cl::Event> mapEvents(num_splits),
        unmapEvents(num_splits),
        decompressEvents(num_splits);
    std::vector<cl::UserEvent> receiveEvents;
    std::vector<void *> ptrs(num_splits);

    /* Setup */
    for (size_t i = 0; i < num_splits + NUM_BUFFERS; i++) {
        const auto &wb = workBuffers[i % NUM_BUFFERS];

        if (i >= NUM_BUFFERS) {
            size_t split_offset = (i - NUM_BUFFERS) * CL_UPLOAD_BLOCK_SIZE;
            size_t split_size = std::min(size - split_offset, CL_UPLOAD_BLOCK_SIZE);

            /* Enqueue unmap buffer (implicit upload) */
            cl::vector<cl::Event> unmapWaitList = {receiveEvents[i - NUM_BUFFERS]};
            commandQueue.enqueueUnmapMemObject(wb, ptrs[i - NUM_BUFFERS],
                                               &unmapWaitList,
                                               &unmapEvents[i - NUM_BUFFERS]);

            // Uncompress (full) compressed blocks
            size_t fullBlocksSize = split_size & ~(lib842::stream::BLOCK_SIZE - 1);

            decompressEvents[i - NUM_BUFFERS] = unmapEvents[i - NUM_BUFFERS];

            if (fullBlocksSize > 0) {
                cl::vector<cl::Event> decompressWaitList = {decompressEvents[i - NUM_BUFFERS]};
                clDataTransferContext.cl842DeviceDecompressor()->decompress(
                    commandQueue, fullBlocksSize / CHUNK_SIZE,
                    wb, 0, cl::Buffer(nullptr),
                    buffer, offset + split_offset, cl::Buffer(nullptr),
                    cl::Buffer(nullptr), cl::Buffer(nullptr),
                    &decompressWaitList, &decompressEvents[i - NUM_BUFFERS]);
            }

            // Partial blocks are not compressed, so we also need to move them from
            // the temporary buffer to the final buffer if necessary
            size_t partialBlockSize = split_size & (lib842::stream::BLOCK_SIZE - 1);
            if (partialBlockSize > 0) {
                cl::vector<cl::Event> decompressWaitList = {decompressEvents[i - NUM_BUFFERS]};
                commandQueue.enqueueCopyBuffer(wb, buffer,
                                               fullBlocksSize, offset + split_offset + fullBlocksSize, partialBlockSize,
                                               &decompressWaitList, &decompressEvents[i - NUM_BUFFERS]);
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
                // On NVIDIA, it appears that enqueuing two map buffer commands on the same buffer,
                // but with two different mapped sizes, leads to unexpected effects
                // It seems that immediately when the second command is (merely) enqueued, it will
                // immediately invalidate the mapping from the first command,
                // making accesses through the not-yet unmapped first pointer crash
                //
                // I can't find anything on the OpenCL standard, and it works on Intel and pocl
                // As a workaround, always map the entire size of the auxiliary buffer
                // TODO: Does this have any performance impact? If so, any workaround / fix?
                0, CL_UPLOAD_BLOCK_SIZE /* Instead of: split_size */,
                eventWaitList, &mapEvents[i]);

            // schedule local data transfer
            cl::UserEvent receiveEvent(clDataTransferContext.context());
            auto mapDataCompletable(std::make_shared<dcl::CLEventCompletable>(mapEvents[i]));
            read(splitTransferIds[i], split_size, ptrs[i], true,
                splitTransferIds[i + 1], split_offset, mapDataCompletable)
                ->setCallback([receiveEvent](cl_int status) mutable { receiveEvent.setStatus(status); });
            receiveEvents.push_back(receiveEvent);
        }
    }

    *startEvent = mapEvents.front();
    *endEvent = decompressEvents.back();
}

void InputDataStream::readToClBufferWithClInplaceDecompression(
        dcl::transfer_id transferId,
        size_t size,
        const dcl::CLInDataTransferContext &clDataTransferContext,
        const cl::Buffer &buffer,
        size_t offset,
        const cl::vector<cl::Event> *eventWaitList,
        cl::Event *startEvent,
        cl::Event *endEvent) {
    auto commandQueue = clDataTransferContext.commandQueue();
    size_t num_splits = (size + CL_UPLOAD_BLOCK_SIZE - 1) / CL_UPLOAD_BLOCK_SIZE;
    auto splitTransferIds = generateClSplitTransferIds(transferId, num_splits);

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
            offset + split_offset, split_size,
            eventWaitList, &mapEvents[i]);
    }

    for (size_t i = 0; i < num_splits; i++) {
        size_t split_offset = i * CL_UPLOAD_BLOCK_SIZE;
        size_t split_size = std::min(size - i * CL_UPLOAD_BLOCK_SIZE, CL_UPLOAD_BLOCK_SIZE);

        // schedule local data transfer
        cl::UserEvent receiveEvent(clDataTransferContext.context());
        auto mapDataCompletable(std::make_shared<dcl::CLEventCompletable>(mapEvents[i]));
        read(splitTransferIds[i], split_size, ptrs[i], true,
            splitTransferIds[i + 1], split_offset, mapDataCompletable)
            ->setCallback([receiveEvent](cl_int status) mutable { receiveEvent.setStatus(status); });

        /* Enqueue unmap buffer (implicit upload) */
        cl::vector<cl::Event> unmapWaitList = {receiveEvent};
        commandQueue.enqueueUnmapMemObject(buffer, ptrs[i], &unmapWaitList, &unmapEvents[i]);

        // Uncompress (full) compressed blocks
        size_t fullBlocksSize = split_size & ~(lib842::stream::BLOCK_SIZE - 1);
        if (fullBlocksSize > 0) {
            cl::vector<cl::Event> decompressWaitList = {unmapEvents[i]};
            clDataTransferContext.cl842DeviceDecompressor()->decompress(
                commandQueue, fullBlocksSize / CHUNK_SIZE,
                buffer, offset + split_offset, cl::Buffer(nullptr),
                buffer, offset + split_offset, cl::Buffer(nullptr),
                cl::Buffer(nullptr), cl::Buffer(nullptr),
                &decompressWaitList, &decompressEvents[i]);
        } else {
            decompressEvents[i] = unmapEvents[i];
        }

        // Partial blocks are not compressed, and since we're uncompressing
        // in-place, the data is already in the right place, so all done
    }


    *startEvent = mapEvents.front();
    *endEvent = decompressEvents.back();
}
#endif

} // namespace comm

} // namespace dclasio
