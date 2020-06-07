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
 * \file OutputDataStream.cpp
 *
 * \date 2014-03-07
 * \author Philipp Kegel
 */

#include "OutputDataStream.h"

#include "DataTransferImpl.h"
#include "DataTransferSentinelChecker.h"
#include "DataTransferProfiler.h"

#ifdef IO_LINK_COMPRESSION
#include <lib842/sw.h>
#include <lib842/stream/comp.h>
#endif
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
#include <lib842/hw.h>
#endif
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION)
#include <lib842/cl.h>
#endif

#include <dcl/Completable.h>
#include <dcl/CLEventCompletable.h>
#include <dcl/DCLTypes.h>

#include <dcl/util/Logger.h>

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
using lib842::stream::CHUNK_SIZE;
using lib842::stream::NUM_CHUNKS_PER_BLOCK;
using lib842::stream::BLOCK_SIZE;
#endif

OutputDataStream::OutputDataStream(boost::asio::ip::tcp::socket& socket)
    : _socket(socket), _sending(false) {
#ifdef IO_LINK_COMPRESSION
    if (dcl::is_io_link_compression_enabled()) {
        auto impl842 = get_optsw842_implementation();
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
        if (is_hw_io_link_compression_enabled()) {
            impl842 = get_hw842_implementation();
        }
#endif
        auto num_threads = determine_io_link_compression_num_threads("DCL_IO_LINK_NUM_COMPRESS_THREADS");
        auto thread_policy = !determine_io_link_compression_spread_threads("DCL_IO_LINK_NUM_COMPRESS_SPREAD")
            ? lib842::stream::thread_policy::use_defaults
            : lib842::stream::thread_policy::spread_threads_among_numa_nodes;

        _compress_thread_pool.reset(new lib842::stream::DataCompressionStream(
            *impl842, num_threads, thread_policy,
            []() -> std::ostream& { return dcl::util::Logger << dcl::util::Error; },
            []() -> std::ostream& { return dcl::util::Logger << dcl::util::Debug; }
        ));

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
        // The other end of the stream depends on this ordering requirement for
        // transfers to OpenCL buffers, which are split in multiple parts,
        // in order not cleanly 'cut' the transfers in those parts
        _compress_thread_pool->set_offset_sync_epoch_multiple(CL_UPLOAD_BLOCK_SIZE);
#endif
    }
#endif
}

std::shared_ptr<DataSending> OutputDataStream::write(
        dcl::transfer_id transfer_id,
        size_t size, const void *ptr, bool skip_compress_step,
        const std::shared_ptr<dcl::Completable> &trigger_event) {
    auto write(std::make_shared<DataSending>(transfer_id, size, ptr, skip_compress_step, dcl::transfer_id(), 0));
    if (trigger_event != nullptr) {
        // If a event to wait was given, enqueue the write after the event callback
        trigger_event->setCallback([this, write](cl_int status) {
            if (status != CL_COMPLETE) {
                dcl::util::Logger << dcl::util::Error
                        << "OutputDataStream: Wait for event (completable) for write enqueue failed"
                        << std::endl;
                write->onFinish(boost::system::errc::make_error_code(boost::system::errc::io_error),
                                0);
                return;
            }
            enqueue_write(write);
        });
    } else {
        // If a event to wait was given, enqueue the write immediately
        enqueue_write(write);
    }
    return write;
}

void OutputDataStream::enqueue_write(const std::shared_ptr<DataSending> &write) {
    std::unique_lock<std::mutex> lock(_writeq_mtx);
    if (_sending) {
        _writeq.push(write);
    } else {
        // start write loop
        _sending = true;
        lock.unlock();

        notify_write_transfer_id(new writeq_type({write}));
    }
}

void OutputDataStream::notify_write_transfer_id(writeq_type *writeq) {
    /* TODO Pass writeq by rvalue reference rather than by pointer
     * is currently not supported by lambdas (see comment in start_write) */
    assert(writeq); // ouch!
    if (writeq->empty()) {
        // pick new writes from the data stream's write queue
        std::lock_guard<std::mutex> lock(_writeq_mtx);
        if (_writeq.empty()) {
            _sending = false;
            delete writeq;
            return; // no more writes - exit write loop
        }
        _writeq.swap(*writeq);
    }
    // writeq is non-empty now

    auto& write = writeq->front();

    boost_asio_async_write_with_sentinels(
        _socket, boost::asio::buffer(write->transferId().data, write->transferId().size()),
        [this, writeq, write](const boost::system::error_code& ec, size_t bytes_transferred) {
            assert(!ec); // TODO: How to handle an error here?
            start_write(writeq);
        });
}

void OutputDataStream::start_write(writeq_type *writeq) {
    auto& write = writeq->front();
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(OutputDataStream to " << _socket.remote_endpoint() << ") "
        << "Start write of size " << write->size()
        << std::endl;
#endif
    write->onStart();
    /* TODO *Move* writeq through (i.e., into and out of) lambda capture
     * In C++14 this should be possible by generalized lambda captures as follows:
    boost_asio_async_write_with_sentinels(
            _socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq{std::move(writeq)}](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(std::move(writeq), ec, bytes_transferred); });
     */

#ifdef IO_LINK_COMPRESSION
    if (dcl::is_io_link_compression_enabled() && write->size() >= BLOCK_SIZE) {
        _write_io_compression_error = false;
        _write_io_channel_busy = false;
        _write_io_total_bytes_transferred = 0;
        _write_io_num_blocks_remaining = write->size() / BLOCK_SIZE;

        if (!write->skip_compress_step()) {
            _compress_thread_pool->start(write->ptr(), write->size(),
                [this, writeq, write](lib842::stream::Block &&block) {
                {
                    std::lock_guard<std::mutex> lock(_write_io_queue_mutex);
                    if (block.offset == SIZE_MAX)
                        _write_io_compression_error = true;
                    if (!_write_io_compression_error)
                        _write_io_queue.push(std::move(block));
                }

                try_write_next_compressed_block_from_compression(writeq, write);
            });
        } else {
            write_next_compressed_block_skip_compression_step(writeq, write);
        }
        return;
    }
#endif

    boost_asio_async_write_with_sentinels(
            _socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(writeq, ec, bytes_transferred); });
}

#ifdef IO_LINK_COMPRESSION
void OutputDataStream::try_write_next_compressed_block_from_compression(writeq_type *writeq, const std::shared_ptr<DataSending> &write) {
    std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
    if (_write_io_channel_busy) {
        // We're already inside a boost::asio::async_write call, so we can't initiate another one until it finishes
        return;
    }
    if (_write_io_num_blocks_remaining == SIZE_MAX) {
        // Last block was already written (calls to this function by compression threads can be spurious)
        return;
    }
    if (_write_io_compression_error) {
        _write_io_num_blocks_remaining = SIZE_MAX;
        lock.unlock();

        _compress_thread_pool->finalize(true, [this, writeq](bool) {
            handle_write(writeq, boost::system::errc::make_error_code(boost::system::errc::io_error),
                         _write_io_total_bytes_transferred);
        });
        return;
    }

    if (_write_io_num_blocks_remaining > 0) {
        if (_write_io_queue.empty()) {
            // No compressed block is yet available
            return;
        }

        const auto &block = _write_io_queue.front();
        _write_io_channel_busy = true;
        lock.unlock();

        // Chunk I/O
        std::array<boost::asio::const_buffer, 2 + NUM_CHUNKS_PER_BLOCK> send_buffers;
        send_buffers[0] = boost::asio::buffer(&block.offset, sizeof(size_t));
        send_buffers[1] = boost::asio::buffer(&block.sizes, sizeof(size_t) * NUM_CHUNKS_PER_BLOCK);
        for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++)
            send_buffers[2 + i] = boost::asio::buffer(block.datas[i], block.sizes[i]);
        boost_asio_async_write_with_sentinels(_socket, send_buffers,
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
             _write_io_channel_busy = false;
             _write_io_queue.pop();
             _write_io_total_bytes_transferred += bytes_transferred;
             _write_io_num_blocks_remaining--;
             lock.unlock();

             if (ec) {
                 _compress_thread_pool->finalize(true, [this, writeq](bool) {
                     handle_write(writeq, boost::system::errc::make_error_code(boost::system::errc::io_error),
                                  _write_io_total_bytes_transferred);
                 });
                 return;
             }

             try_write_next_compressed_block_from_compression(writeq, write);
         });
    } else {
        // Always write the last incomplete block of the input uncompressed
        _write_io_channel_busy = true;
        lock.unlock();

        auto last_block_source_ptr =  static_cast<const uint8_t *>(write->ptr()) + (write->size() & ~(BLOCK_SIZE - 1));
        auto last_block_size = write->size() & (BLOCK_SIZE - 1);

        boost_asio_async_write_with_sentinels(_socket,
                boost::asio::buffer(last_block_source_ptr, last_block_size),
            [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
            {
                std::lock_guard<std::mutex> lock(_write_io_queue_mutex);
                _write_io_channel_busy = false;
                _write_io_total_bytes_transferred += bytes_transferred;
                _write_io_num_blocks_remaining = SIZE_MAX;
            }

            _compress_thread_pool->finalize(false, [this, writeq, ec](bool) {
                handle_write(writeq, ec, _write_io_total_bytes_transferred);
            });
        });
    }
}

void OutputDataStream::write_next_compressed_block_skip_compression_step(writeq_type *writeq, const std::shared_ptr<DataSending> &write) {
    if (_write_io_num_blocks_remaining > 0) {
        // Create block using already compressed data
        size_t blockno = write->size() / BLOCK_SIZE - _write_io_num_blocks_remaining;
        _write_io_block_scs.offset = blockno * BLOCK_SIZE;
        for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++) {
            auto source = static_cast<const uint8_t *>(write->ptr()) + _write_io_block_scs.offset + i * CHUNK_SIZE;

            auto is_compressed = std::equal(source,source + sizeof(LIB842_COMPRESSED_CHUNK_MARKER), LIB842_COMPRESSED_CHUNK_MARKER);

            auto chunk_buffer_size = is_compressed
                 ? *reinterpret_cast<const uint64_t *>((source + sizeof(LIB842_COMPRESSED_CHUNK_MARKER)))
                : CHUNK_SIZE;
            auto chunk_buffer = is_compressed
                    ? source + CHUNK_SIZE - chunk_buffer_size
                    : source;

            _write_io_block_scs.datas[i] = chunk_buffer;
            _write_io_block_scs.sizes[i] = chunk_buffer_size;
        }

        // Chunk I/O
        std::array<boost::asio::const_buffer, 2 + NUM_CHUNKS_PER_BLOCK> send_buffers;
        send_buffers[0] = boost::asio::buffer(&_write_io_block_scs.offset, sizeof(size_t));
        send_buffers[1] = boost::asio::buffer(&_write_io_block_scs.sizes, sizeof(size_t) * NUM_CHUNKS_PER_BLOCK);
        for (size_t i = 0; i < NUM_CHUNKS_PER_BLOCK; i++)
            send_buffers[2 + i] = boost::asio::buffer(_write_io_block_scs.datas[i], _write_io_block_scs.sizes[i]);
        boost_asio_async_write_with_sentinels(_socket, send_buffers,
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             _write_io_total_bytes_transferred += bytes_transferred;
             _write_io_num_blocks_remaining--;

             if (ec) {
                 handle_write(writeq, boost::system::errc::make_error_code(boost::system::errc::io_error),
                              _write_io_total_bytes_transferred);
                 return;
             }

             write_next_compressed_block_skip_compression_step(writeq, write);
         });
    } else {
        // Always write the last incomplete block of the input uncompressed
        auto last_block_source_ptr =  static_cast<const uint8_t *>(write->ptr()) + (write->size() & ~(BLOCK_SIZE - 1));
        auto last_block_size = write->size() & (BLOCK_SIZE - 1);

        boost_asio_async_write_with_sentinels(_socket,
                boost::asio::buffer(last_block_source_ptr, last_block_size),
            [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
            _write_io_total_bytes_transferred += bytes_transferred;
            _write_io_num_blocks_remaining = SIZE_MAX;

            handle_write(writeq, ec, _write_io_total_bytes_transferred);
        });
    }
}
#endif

void OutputDataStream::handle_write(
        writeq_type *writeq,
        const boost::system::error_code& ec,
        size_t bytes_transferred) {
    // current write is first element in writeq, so writeq must be non-empty
    assert(writeq /* ouch! */ && !writeq->empty());
    auto& write = writeq->front();
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(OutputDataStream to " << _socket.remote_endpoint() << ") "
        << "End write of size " << write->size()
        << std::endl;
#endif

    write->onFinish(ec, bytes_transferred);
    writeq->pop();

    if (ec) {
        // TODO Handle errors
    }

    notify_write_transfer_id(writeq); // process remaining writes
}

void OutputDataStream::writeFromClBuffer(
        dcl::transfer_id transferId,
        size_t size,
        const dcl::CLOutDataTransferContext &clDataTransferContext,
        const cl::Buffer &buffer,
        size_t offset,
        const cl::vector<cl::Event> *eventWaitList,
        cl::Event *startEvent,
        cl::Event *endEvent) {
    cl::Event myStartEvent;
    if (startEvent == nullptr)
        startEvent = &myStartEvent;

    cl::UserEvent sendData(clDataTransferContext.context());

    /* Enqueue map buffer */
    void *ptr = clDataTransferContext.commandQueue().enqueueMapBuffer(
        buffer,
        CL_FALSE,     // non-blocking map
        CL_MAP_READ, // map for reading
        offset, size,
        eventWaitList, startEvent);
    // schedule local data transfer
    auto mapDataCompletable(std::make_shared<dcl::CLEventCompletable>(*startEvent));
    write(transferId, size, ptr, false, mapDataCompletable)
        ->setCallback([sendData](cl_int status) mutable { sendData.setStatus(status); });
    /* Enqueue unmap buffer (implicit upload) */
    cl::vector<cl::Event> unmapWaitList = {sendData};
    clDataTransferContext.commandQueue().enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);

    profile_transfer(profile_transfer_direction::send, transferId, size, *startEvent, *endEvent);
}

} // namespace comm

} // namespace dclasio
