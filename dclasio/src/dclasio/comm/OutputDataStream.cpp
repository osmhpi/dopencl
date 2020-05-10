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

#include <sw842.h>
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
#include <hw842.h>
#endif
#include <compstream842.h>

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
// Declarations for static constexpr are sometimes required to avoid build errors
// See https://stackoverflow.com/questions/8016780/undefined-reference-to-static-constexpr-char
constexpr size_t OutputDataStream::NUM_CHUNKS_PER_NETWORK_BLOCK;
constexpr size_t OutputDataStream::CHUNK_SIZE;
constexpr size_t OutputDataStream::NETWORK_BLOCK_SIZE;
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
constexpr size_t OutputDataStream::CL_UPLOAD_BLOCK_SIZE;
#endif
#endif

OutputDataStream::OutputDataStream(boost::asio::ip::tcp::socket& socket)
    : _socket(socket), _sending(false) {
#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        auto compress842_func = optsw842_compress;
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
        if (is_hw_io_link_compression_enabled())
            compress842_func = hw842_compress;
#endif

        _compress_thread_pool.reset(new lib842::stream::DataCompressionStream(
            compress842_func,
            determine_io_link_compression_num_threads("DCL_IO_LINK_NUM_COMPRESS_THREADS"),
            []() -> std::ostream& { return dcl::util::Logger << dcl::util::Error; },
            []() -> std::ostream& { return dcl::util::Logger << dcl::util::Debug; }
        ));
    }
#endif
}

std::shared_ptr<DataSending> OutputDataStream::write(
        dcl::transfer_id transfer_id,
        size_t size, const void *ptr, bool skip_compress_step,
        const std::shared_ptr<dcl::Completable> &trigger_event) {
    auto write(std::make_shared<DataSending>(transfer_id, size, ptr, skip_compress_step));
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
    std::vector<std::shared_ptr<DataSending>> writes = {write};

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    if (is_io_link_compression_enabled() && is_cl_io_link_compression_enabled() &&
        write->size() > CL_UPLOAD_BLOCK_SIZE) {
        // TODOXXX: The OpenCL-based decompression code does work in blocks of
        //          size CL_UPLOAD_BLOCK_SIZE, and for now, it splits its reads
        //          in blocks of this size. However, in order for all read/writes
        //          to match correctly, *all* transfers need to be split in blocks
        //          of this size. This is a huge hack and OpenCL-based decompression
        //          should eventually be integrated better so this isn't necessary
        writes.clear();

        size_t num_splits = (write->size() + CL_UPLOAD_BLOCK_SIZE - 1) / CL_UPLOAD_BLOCK_SIZE;
        dcl::transfer_id split_transfer_id = write->transferId();
        for (size_t i = 0; i < num_splits; i++, dcl::next_cl_split_transfer_id(split_transfer_id)) {
            size_t split_offset = i * CL_UPLOAD_BLOCK_SIZE;
            size_t split_size = std::min(write->size() - split_offset, CL_UPLOAD_BLOCK_SIZE);
            auto subwrite = std::make_shared<DataSending>(
                split_transfer_id, split_size,
                static_cast<const uint8_t *>(write->ptr()) + split_offset,
                write->skip_compress_step());
            writes.push_back(subwrite);
        }

        // Complete the main (non-split) transfer when the last split part finishes
        // TODOXXX error handling is not at all solid here
        writes.back()->setCallback([write](cl_int status) {
            auto ec = status == CL_SUCCESS
                 ? boost::system::error_code()
                 : boost::system::errc::make_error_code(boost::system::errc::io_error);
            write->onFinish(ec, write->size());
        });
    }
#endif

    std::unique_lock<std::mutex> lock(_writeq_mtx);
    if (_sending) {
        for (const auto &w : writes)
            _writeq.push(w);
    } else {
        // start write loop
        _sending = true;
        lock.unlock();

        notify_write_transfer_id(new writeq_type(
            std::list<std::shared_ptr<DataSending>>(writes.begin(), writes.end())));
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
    if (is_io_link_compression_enabled()) {
        _write_io_compression_error = false;
        _write_io_channel_busy = false;
        _write_io_total_bytes_transferred = 0;
        _write_io_num_blocks_remaining = write->size() / NETWORK_BLOCK_SIZE;

        if (write->size() >= NETWORK_BLOCK_SIZE) {
            _compress_thread_pool->start(
                write->ptr(), write->size(), write->skip_compress_step(),
                [this, writeq, write](lib842::stream::DataCompressionStream::compress_block &&block) {
                {
                    std::lock_guard<std::mutex> lock(_write_io_queue_mutex);
                    if (block.source_offset == SIZE_MAX)
                        _write_io_compression_error = true;
                    if (!_write_io_compression_error)
                        _write_io_queue.push(std::move(block));
                }

                try_write_next_compressed_block(writeq, write);
            });
        } else {
            try_write_next_compressed_block(writeq, write);
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
void OutputDataStream::try_write_next_compressed_block(writeq_type *writeq, const std::shared_ptr<DataSending> &write) {
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

        assert(write->size() >= NETWORK_BLOCK_SIZE);
        _compress_thread_pool->finish(false);
        handle_write(writeq, boost::system::errc::make_error_code(boost::system::errc::io_error),
                     _write_io_total_bytes_transferred);
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
        std::array<boost::asio::const_buffer, 2 + NUM_CHUNKS_PER_NETWORK_BLOCK> send_buffers;
        send_buffers[0] = boost::asio::buffer(&block.source_offset, sizeof(size_t));
        send_buffers[1] = boost::asio::buffer(&block.sizes, sizeof(size_t) * NUM_CHUNKS_PER_NETWORK_BLOCK);
        for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++)
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
                 assert(write->size() >= NETWORK_BLOCK_SIZE);
                 _compress_thread_pool->finish(true);
                 handle_write(writeq, boost::system::errc::make_error_code(boost::system::errc::io_error),
                              _write_io_total_bytes_transferred);
                 return;
             }

             try_write_next_compressed_block(writeq, write);
         });
    } else {
        // Always write the last incomplete block of the input uncompressed
        _write_io_channel_busy = true;
        lock.unlock();

        auto last_block_source_ptr =  static_cast<const uint8_t *>(write->ptr()) + (write->size() & ~(NETWORK_BLOCK_SIZE - 1));
        auto last_block_size = write->size() & (NETWORK_BLOCK_SIZE - 1);

        boost_asio_async_write_with_sentinels(_socket,
                boost::asio::buffer(last_block_source_ptr, last_block_size),
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             {
                 std::lock_guard<std::mutex> lock(_write_io_queue_mutex);
                 _write_io_channel_busy = false;
                 _write_io_total_bytes_transferred += bytes_transferred;
                 _write_io_num_blocks_remaining = SIZE_MAX;
             }

             // The data transfer thread also joins the final barrier for the compression
             // threads before finishing the write, to ensure resources are not released
             // while a compression thread still hasn't realized all work is finished
             if (write->size() >= NETWORK_BLOCK_SIZE)
                _compress_thread_pool->finish(false);

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
        const cl::Context &context,
        const cl::CommandQueue &commandQueue,
        const cl::Buffer &buffer,
        size_t offset,
        const cl::vector<cl::Event> *eventWaitList,
        cl::Event *startEvent,
        cl::Event *endEvent) {
    cl::Event myStartEvent;
    if (startEvent == nullptr)
        startEvent = &myStartEvent;

    cl::UserEvent sendData(context);

    /* Enqueue map buffer */
    void *ptr = commandQueue.enqueueMapBuffer(
        buffer,
        CL_FALSE,     // non-blocking map
        CL_MAP_READ, // map for reading
        offset, size,
        eventWaitList, startEvent);
    // schedule local data transfer
    std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(*startEvent));
    write(transferId, size, ptr, false, mapDataCompletable)
        ->setCallback(std::bind(&cl::UserEvent::setStatus, sendData, std::placeholders::_1));
    /* Enqueue unmap buffer (implicit upload) */
    cl::vector<cl::Event> unmapWaitList = {sendData};
    commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);

    profile_transfer(profile_transfer_direction::send, transferId, size, *startEvent, *endEvent);
}

} // namespace comm

} // namespace dclasio
