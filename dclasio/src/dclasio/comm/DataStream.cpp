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
 * \file DataStream.cpp
 *
 * \date 2014-03-07
 * \author Philipp Kegel
 */

#include "DataStream.h"

#include "DataTransferImpl.h"

#include <dcl/ByteBuffer.h>
#include <dcl/Completable.h>
#include <dcl/CLEventCompletable.h>
#include <dcl/DCLTypes.h>

#include <dcl/util/Logger.h>

#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

#define PROFILE_SEND_RECEIVE_BUFFER

#ifdef PROFILE_SEND_RECEIVE_BUFFER
#include <chrono>

struct profile_send_receive_buffer_times {
    dcl::transfer_id id;
    size_t transfer_size;
    std::chrono::time_point<std::chrono::steady_clock> enqueue_time;
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    std::chrono::time_point<std::chrono::steady_clock> end_time;
};
#endif

// TODOXXX: This class has grown way too large and has way too many responsabilities
// It should be split into multiple files/classes like:
// * Receiving DataStream
// * Sending DataStream
// * Decompression pool
// * Compression pool
// * Boost.Asio sentinel helper

#ifdef IO_LINK_COMPRESSION

// Declarations for static constexpr are sometimes required to avoid build errors
// See https://stackoverflow.com/questions/8016780/undefined-reference-to-static-constexpr-char
constexpr size_t dclasio::comm::InputDataStream::NUM_CHUNKS_PER_NETWORK_BLOCK;
constexpr size_t dclasio::comm::OutputDataStream::NUM_CHUNKS_PER_NETWORK_BLOCK;
constexpr size_t dclasio::comm::InputDataStream::CHUNK_SIZE;
constexpr size_t dclasio::comm::OutputDataStream::CHUNK_SIZE;
constexpr size_t dclasio::comm::InputDataStream::NETWORK_BLOCK_SIZE;
constexpr size_t dclasio::comm::OutputDataStream::NETWORK_BLOCK_SIZE;
constexpr size_t dclasio::comm::InputDataStream::COMPRESSIBLE_THRESHOLD;
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
constexpr size_t dclasio::comm::InputDataStream::CL_UPLOAD_BLOCK_SIZE;
constexpr size_t dclasio::comm::OutputDataStream::CL_UPLOAD_BLOCK_SIZE;
#endif

// TODOXXX: This is a variation of next_cl_split_transfer_id,
//          but used to get some hacky code to work, see call sites. Remove me.
static void next_cl_split_transfer_id(dcl::transfer_id &transfer_id) {
    for (size_t i = 4; i < boost::uuids::uuid::static_size(); i++) {
        transfer_id.data[boost::uuids::uuid::static_size()-i-1]++;
        if (transfer_id.data[boost::uuids::uuid::static_size()-i-1] != 0)
            break;
    }
}

#endif

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

// If USE_SENTINELS is defined, special marker sequences are included and checked before/after each data transfer
// If marker sequences in a read operation don't match those in each write operation, an assertion will be triggered
// This is useful as a "fail fast" for stream desynchronization, for bugs such as race conditions
//#define USE_SENTINELS

#ifdef USE_SENTINELS
static constexpr unsigned int SENTINEL_START = 0x12345678, SENTINEL_END = 0x87654321;
#endif

/**
 * Like boost::asio::async_write, but also sends sentinels
 * through the stream if USE_SENTINELS is set.
 */
template<typename AsyncWriteStream, typename ConstBufferSequence, typename WriteHandler>
static auto boost_asio_async_write_with_sentinels(AsyncWriteStream &s, const ConstBufferSequence &buffers, WriteHandler &&handler)
    -> decltype(boost::asio::async_write(s, buffers, handler)) {
#ifdef USE_SENTINELS
    std::vector<boost::asio::const_buffer> buffers_with_sentinels;
    for (const auto &b : buffers) {
        buffers_with_sentinels.push_back(boost::asio::buffer(&SENTINEL_START, sizeof(SENTINEL_START)));
        buffers_with_sentinels.push_back(b);
        buffers_with_sentinels.push_back(boost::asio::buffer(&SENTINEL_END, sizeof(SENTINEL_END)));
    }
    return boost::asio::async_write(s, buffers_with_sentinels, handler);
#else
    return boost::asio::async_write(s, buffers, handler);
#endif
}

/**
 * Like boost::asio::async_read, but also reads and verifies sentinels
 * through the stream if USE_SENTINELS is set.
 */
template<typename AsyncReadStream, typename MutableBufferSequence, typename ReadHandler>
static auto boost_asio_async_read_with_sentinels(AsyncReadStream &s, const MutableBufferSequence &buffers, ReadHandler &&handler)
    -> decltype(boost::asio::async_read(s, buffers, handler)) {
#ifdef USE_SENTINELS
    struct sentinel_t {
        unsigned int start, end;
    };

    // Since the sentinels need to survive the asynchronous read call, we need to use a
    // std::shared_ptr for the sentinels to survive until the read completes
    // TODO: This can be a std::unique_ptr in C++14 with lambda generalized capture
    //       https://stackoverflow.com/a/16968463
    auto sentinels(std::make_shared<std::vector<sentinel_t>>(
            std::distance(buffers.begin(), buffers.end())));

    size_t i = 0;
    std::vector<boost::asio::mutable_buffer> buffers_with_sentinels;
    for (const auto &b : buffers) {
        buffers_with_sentinels.push_back(boost::asio::buffer(&(*sentinels)[i].start, sizeof(sentinel_t::start)));
        buffers_with_sentinels.push_back(b);
        buffers_with_sentinels.push_back(boost::asio::buffer(&(*sentinels)[i].end, sizeof(sentinel_t::end)));
        i++;
    }

    return boost::asio::async_read(
            s, buffers_with_sentinels,
            [handler, sentinels](const boost::system::error_code &ec, size_t bytes_transferred) {
                if (!ec) { // On error, we expect the original handler to handle the failure
                    if (!std::all_of(sentinels->begin(), sentinels->end(),
                        [](const sentinel_t &s) {
                            return s.start == SENTINEL_START && s.end == SENTINEL_END;
                        })) {
                        dcl::util::Logger << dcl::util::Error
                                << "DataStream: Mismatched read/write calls (sentinel mismatch)"
                                << std::endl;
                        handler(boost::system::errc::make_error_code(boost::system::errc::io_error),
                                bytes_transferred);
                        return;
                    }
                }
                handler(ec, bytes_transferred);
            });
#else
    return boost::asio::async_read(s, buffers, handler);
#endif
}

namespace dclasio {

namespace comm {

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket) :
        _socket(socket),
        InputDataStream(*socket), OutputDataStream(*socket)
{
    // TODO Ensure that socket is connected
    _remote_endpoint = _socket->remote_endpoint();
}

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket,
        boost::asio::ip::tcp::endpoint remote_endpoint) :
        _socket(socket), _remote_endpoint(remote_endpoint),
        InputDataStream(*socket), OutputDataStream(*socket)
{
    assert(!socket->is_open()); // socket must not be connect
}

dcl::process_id DataStream::connect(
        dcl::process_id pid) {
    _socket->connect(_remote_endpoint); // connect socket to remote endpoint

    // send process ID to remote process via data stream
    // TODO Encode local process type and data stream protocol
    dcl::OutputByteBuffer buf;
    buf << pid << uint8_t(0) << uint8_t(0);
    boost::asio::write(*_socket, boost::asio::buffer(buf.data(), buf.size()));
    dcl::util::Logger << dcl::util::Verbose
            << "Sent process identification message for data stream (pid=" << pid << ')'
            << std::endl;

#if USE_DATA_STREAM_RESPONSE
    // receive response
    dcl::InputByteBuffer ibuf;
    ibuf.resize(sizeof(dcl::process_id));
    boost::asio::read(*_socket, boost::asio::buffer(ibuf.data(), ibuf.size()));
    ibuf >> pid;
    dcl::util::Logger << dcl::util::Verbose
            << "Received identification message response (pid=" << pid << ')'
            << std::endl;
#endif

    return pid;
}

void DataStream::disconnect() {
    _socket->shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    _socket->close();
}

InputDataStream::InputDataStream(boost::asio::ip::tcp::socket& socket)
    : _socket(socket), _read_state(receiving_state::idle) {
#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        _decompress_thread_pool.reset(new DataDecompressionWorkPool());
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
    if (is_io_link_compression_enabled() && is_cl_io_link_compression_enabled() &&
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
        for (size_t i = 0; i < num_splits; i++, next_cl_split_transfer_id(split_transfer_id)) {
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
    if (is_io_link_compression_enabled()) {
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
                        std::copy(CL842_COMPRESSED_CHUNK_MAGIC, CL842_COMPRESSED_CHUNK_MAGIC + sizeof(CL842_COMPRESSED_CHUNK_MAGIC), destination);
                        *reinterpret_cast<uint64_t *>((destination + sizeof(CL842_COMPRESSED_CHUNK_MAGIC))) = _read_io_buffer_sizes[i];
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
                DataDecompressionWorkPool::decompress_block dm;
                dm.compress_buffer.reset(_read_io_compressed_buffer.release());
                bool should_uncompress_any = false;
                for (size_t i = 0, compressed_buffer_offset = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                    if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && !_read_op->skip_compress_step()) {
                        dm.chunks[i] = DataDecompressionWorkPool::decompress_chunk{
                            .compressed_data = dm.compress_buffer.get() + compressed_buffer_offset,
                            .compressed_length = _read_io_buffer_sizes[i],
                            .destination = static_cast<uint8_t *>(_read_op->ptr()) + _read_io_destination_offset + i * CHUNK_SIZE
                        };
                        compressed_buffer_offset += _read_io_buffer_sizes[i];
                        should_uncompress_any = true;
                    } else {
                        dm.chunks[i] = DataDecompressionWorkPool::decompress_chunk{
                            .compressed_data = nullptr,
                            .compressed_length = 0,
                            .destination = nullptr
                        };
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
        const CL842DeviceDecompressor *cl842DeviceDecompressor,
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

    if (is_io_link_compression_enabled() && is_cl_io_link_compression_enabled() &&
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
                size_t chunksSize = split_size & ~(dcl::DataTransfer::COMPR842_CHUNK_SIZE - 1);
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

                next_cl_split_transfer_id(split_transfer_id);
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
        for (size_t i = 0; i < num_splits; i++, next_cl_split_transfer_id(split_transfer_id)) {
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
            size_t chunksSize = split_size & ~(dcl::DataTransfer::COMPR842_CHUNK_SIZE - 1);
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

#ifdef PROFILE_SEND_RECEIVE_BUFFER
    auto profile_times = new profile_send_receive_buffer_times();
    try {
        profile_times->id = transferId;
        profile_times->transfer_size = size;
        profile_times->enqueue_time = std::chrono::steady_clock::now();

        startEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
            auto profile_times = ((profile_send_receive_buffer_times *)user_data);
            profile_times->start_time = std::chrono::steady_clock::now();
            dcl::util::Logger << dcl::util::Debug
                              << "(PROFILE) Receive with id " << profile_times->id << " of size " << profile_times->transfer_size
                              << " started (ENQUEUE -> START) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                profile_times->start_time - profile_times->enqueue_time).count() << std::endl;
        }, profile_times);

        endEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
            std::unique_ptr<profile_send_receive_buffer_times> profile_times(
                        static_cast<profile_send_receive_buffer_times *>(user_data));
            profile_times->end_time = std::chrono::steady_clock::now();
            dcl::util::Logger << dcl::util::Debug
                              << "(PROFILE) Receive with id " << profile_times->id << " of size " << profile_times->transfer_size
                              << " uploaded (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                profile_times->end_time - profile_times->start_time).count() << std::endl;
        }, profile_times);
    } catch (...) {
        delete profile_times;
        throw;
    }
#endif
}

OutputDataStream::OutputDataStream(boost::asio::ip::tcp::socket& socket)
    : _socket(socket), _sending(false) {
#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        _compress_thread_pool.reset(new DataCompressionWorkPool());
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
        for (size_t i = 0; i < num_splits; i++, next_cl_split_transfer_id(split_transfer_id)) {
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
                [this, writeq, write](DataCompressionWorkPool::compress_block &&block) {
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

#ifdef PROFILE_SEND_RECEIVE_BUFFER
    auto profile_times = new profile_send_receive_buffer_times();
    try {
        profile_times->id = transferId;
        profile_times->transfer_size = size;
        profile_times->enqueue_time = std::chrono::steady_clock::now();

        startEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
            auto profile_times = ((profile_send_receive_buffer_times *)user_data);
            profile_times->start_time = std::chrono::steady_clock::now();
            dcl::util::Logger << dcl::util::Debug
                << "(PROFILE) Send with id " << profile_times->id << " of size " << profile_times->transfer_size
                << " started (ENQUEUE -> START) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                        profile_times->start_time - profile_times->enqueue_time).count() << std::endl;
        }, profile_times);

        endEvent->setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
            std::unique_ptr<profile_send_receive_buffer_times> profile_times(
                        static_cast<profile_send_receive_buffer_times *>(user_data));
            profile_times->end_time = std::chrono::steady_clock::now();
            dcl::util::Logger << dcl::util::Debug
                << "(PROFILE) Send with id " << profile_times->id << " of size " << profile_times->transfer_size
                << " uploaded (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                        profile_times->end_time - profile_times->start_time).count() << std::endl;
        }, profile_times);
    } catch (...) {
        delete profile_times;
        throw;
    }
#endif
}

} // namespace comm

} // namespace dclasio
