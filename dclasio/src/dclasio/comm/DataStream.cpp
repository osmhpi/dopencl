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
#include <dcl/DCLTypes.h>

#include <dcl/util/Logger.h>

#include <boost/asio/buffer.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <boost/asio/ip/tcp.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <mutex>
#include <utility>

#ifdef IO_LINK_COMPRESSION
#define CHUNK_SIZE 16384
static const uint8_t CHUNK_MAGIC[16] = {
        0xbe, 0x5a, 0x46, 0xbf, 0x97, 0xe5, 0x2d, 0xd7, 0xb2, 0x7c, 0x94, 0x1a, 0xee, 0xd6, 0x70, 0x76
};
#define COMPRESSIBLE_THRESHOLD ((CHUNK_SIZE - sizeof(CHUNK_MAGIC) - sizeof(size_t)))

#include <queue>

#ifndef USE_HW_IO_LINK_COMPRESSION
#include <sw842.h>
#define lib842_compress sw842_compress
#define lib842_decompress sw842_decompress
#else
#include <hw842.h>
#define lib842_compress hw842_compress
#define lib842_decompress hw842_decompress
#endif
#endif

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
    std::shared_ptr<std::vector<sentinel_t>> sentinels(new std::vector<sentinel_t>(
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
                    for (const auto &s : *sentinels) {
                        assert(s.start == SENTINEL_START && s.end == SENTINEL_END);
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
        _socket(socket), _receiving(false), _sending(false) {
    // TODO Ensure that socket is connected
    _remote_endpoint = _socket->remote_endpoint();

#ifdef IO_LINK_COMPRESSION
    _decompress_thread = std::thread{&DataStream::start_decompress_thread, this};
    _compress_thread = std::thread{&DataStream::start_compress_thread, this};
#endif
}

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket,
        boost::asio::ip::tcp::endpoint remote_endpoint) :
        _socket(socket), _remote_endpoint(remote_endpoint), _receiving(false), _sending(false) {
    assert(!socket->is_open()); // socket must not be connect

#ifdef IO_LINK_COMPRESSION
    _decompress_thread = std::thread{&DataStream::start_decompress_thread, this};
    _compress_thread = std::thread{&DataStream::start_compress_thread, this};
#endif
}

DataStream::~DataStream() {
#ifdef IO_LINK_COMPRESSION
    {
        std::unique_lock<std::mutex> lock(read_decompress_queue_mutex);
        read_decompress_queue.push(read_queue_quit_message());
    }
    read_decompress_queue_available.notify_one();
    _decompress_thread.join();

    {
        std::unique_lock<std::mutex> lock(write_thread_mutex);
        write_thread_trigger = true;
        write_thread_trigger_quit = true;
    }
    write_thread_trigger_cv.notify_one();
    _compress_thread.join();
#endif
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
    buf.resize(sizeof(dcl::process_id));
    boost::asio::read(*_socket, boost::asio::buffer(buf.data(), buf.size()));
    buf >> pid;
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

std::shared_ptr<DataReceipt> DataStream::read(
        size_t size, void *ptr, bool skip_compress_step) {
    auto read(std::make_shared<DataReceipt>(size, ptr, skip_compress_step));

    std::unique_lock<std::mutex> lock(_readq_mtx);
    if ((_receiving)) {
        _readq.push(read);
    } else {
        // start read loop
        _receiving = true;
        lock.unlock();

        start_read(new readq_type({ read }));
    }

    return read;
}

std::shared_ptr<DataSending> DataStream::write(
        size_t size, const void *ptr, bool skip_compress_step) {
    auto write(std::make_shared<DataSending>(size, ptr, skip_compress_step));

    std::unique_lock<std::mutex> lock(_writeq_mtx);
    if (_sending) {
        _writeq.push(write);
    } else {
        // start write loop
        _sending = true;
        lock.unlock();

        start_write(new writeq_type({ write }));
    }

    return write;
}

void DataStream::start_read(
        readq_type *readq) {
    /* TODO Pass readq by rvalue reference rather than by pointer
     * This is currently not supported by lambdas (see comment in start_write) */
    assert(readq); // ouch!
    if (readq->empty()) {
        // pick new reads from the data stream's read queue
        std::lock_guard<std::mutex> lock(_readq_mtx);
        if (_readq.empty()) {
            _receiving = false;
            delete readq;
            return; // no more reads - exit read loop
        }
        _readq.swap(*readq);
    }
    // readq is non-empty now

    auto& read = readq->front();
    read->onStart();
#ifndef IO_LINK_COMPRESSION
    boost_asio_async_read_with_sentinels(
            *_socket, boost::asio::buffer(read->ptr(), read->size()),
            [this, readq](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_read(readq, ec, bytes_transferred); });
#else
    decompress_inflight = 0;
    rq_cumulative_transfer = 0;
    rq_remaining_offset = 0;

    read_next_compressed_chunk(readq, read);
#endif
}

#ifdef IO_LINK_COMPRESSION
void DataStream::read_next_compressed_chunk(readq_type *readq, std::shared_ptr<DataReceipt> read) {
    if (read->size() - rq_remaining_offset >= CHUNK_SIZE) {
        boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer(&rq_compressed_size, sizeof(size_t)),
                [this, readq, read] (const boost::system::error_code& ec, size_t bytes_transferred){
            assert(!ec);
            rq_cumulative_transfer += bytes_transferred;
            rq_compress_buffer.resize(rq_compressed_size);
            boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer(rq_compress_buffer.data(), rq_compressed_size),
            [this, readq, read] (const boost::system::error_code& ec, size_t bytes_transferred) {
                assert(!ec);
                // Push into the chunk queue
                std::unique_lock<std::mutex> lock(read_decompress_queue_mutex);
                read_queue_decompress_message dm = {
                        .compressed_data = std::move(rq_compress_buffer),
                        .destination = static_cast<uint8_t *>(read->ptr()) + rq_remaining_offset,
                        .skip_compress_step = read->skip_compress_step()
                };
                readq_message_t msg(std::move(dm));
                read_decompress_queue.emplace(std::move(msg));
                lock.unlock();
                read_decompress_queue_available.notify_one();

                rq_cumulative_transfer += bytes_transferred;
                rq_remaining_offset += CHUNK_SIZE;

                read_next_compressed_chunk(readq, read);
            });
        });
    } else {
        // Always read the last incomplete chunk of the input uncompressed
        boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer((uint8_t *) read->ptr() + rq_remaining_offset, read->size() - rq_remaining_offset),
                                [this, readq, read](const boost::system::error_code &ec, size_t bytes_transferred) {
            assert(!ec);
            std::unique_lock<std::mutex> lock(read_decompress_queue_mutex);
            bool done = decompress_inflight == 0 && read_decompress_queue.empty();
            if (!done) {
                read_decompress_queue.push(read_queue_finalize_message {
                    .readq = readq
                });
            }
            read_decompress_queue_available.notify_one();
            lock.unlock();

            rq_cumulative_transfer += bytes_transferred;
            rq_remaining_offset = readq->size();

            if (done) {
                handle_read(readq, boost::system::error_code(), rq_cumulative_transfer);
            }
        });
    }
}

void DataStream::start_decompress_thread() {
    while (true) {
        // (Blocking) pop from the chunk queue
        std::unique_lock<std::mutex> lock(read_decompress_queue_mutex);
        while (read_decompress_queue.empty())
            read_decompress_queue_available.wait(lock);
        auto chunkVariant = std::move(read_decompress_queue.front());
        read_decompress_queue.pop();
        if (chunkVariant.which() == 1) { // read_queue_finalize_message
            auto finalize_message = boost::get<read_queue_finalize_message>(chunkVariant);
            handle_read(finalize_message.readq, boost::system::error_code(), rq_cumulative_transfer);
        } else if (chunkVariant.which() == 2) { // read_queue_quit_message
            break;
        } else if (chunkVariant.which() == 0) { // read_queue_decompress_message
            decompress_inflight++;
            auto chunk = std::move(boost::get<read_queue_decompress_message>(chunkVariant));
            lock.unlock();

            if (chunk.compressed_data.size() <= COMPRESSIBLE_THRESHOLD) {
                if (chunk.skip_compress_step) {
                    std::copy(CHUNK_MAGIC, CHUNK_MAGIC + sizeof(CHUNK_MAGIC), (uint8_t *) chunk.destination);
                    *(size_t *) ((uint8_t *) chunk.destination + sizeof(CHUNK_MAGIC)) = chunk.compressed_data.size();
                    std::copy(chunk.compressed_data.data(), chunk.compressed_data.data() + chunk.compressed_data.size(),
                              (uint8_t *) chunk.destination + sizeof(CHUNK_MAGIC) + sizeof(size_t));
                } else {
                    size_t uncompressed_size = CHUNK_SIZE;
                    int ret = lib842_decompress((const uint8_t *) chunk.compressed_data.data(),
                                                chunk.compressed_data.size(),
                                                (uint8_t *) chunk.destination, &uncompressed_size);
                    assert(ret == 0);
                    assert(uncompressed_size == CHUNK_SIZE);
                }
            } else {
                assert(chunk.compressed_data.size() == CHUNK_SIZE);
                std::copy(chunk.compressed_data.data(), chunk.compressed_data.data() + CHUNK_SIZE,
                          (uint8_t *) chunk.destination);
            }

            lock.lock();
            decompress_inflight--;
            lock.unlock();
        } else {
            assert(0);
        }
    }
}
#endif

void DataStream::handle_read(
        readq_type *readq,
        const boost::system::error_code& ec,
        size_t bytes_transferred) {
    // current read is first element in readq, so readq must be non-empty
    assert(readq /* ouch! */ && !readq->empty());
    readq->front()->onFinish(ec, bytes_transferred);
    readq->pop();

    if (ec) {
        // TODO Handle errors
    }

    start_read(readq); // process remaining reads
}

void DataStream::start_write(
        writeq_type *writeq) {
    /* TODO Pass writeq by rvalue reference rather than by pointer
     * is currently not supported by lambdas (see comment below) */
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
    write->onStart();
    /* TODO *Move* writeq through (i.e., into and out of) lambda capture
     * In C++14 this should be possible by generalized lambda captures as follows:
    boost_asio_async_write_with_sentinels(
            *_socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq{std::move(writeq)}](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(std::move(writeq), ec, bytes_transferred); });
     */
#ifndef IO_LINK_COMPRESSION
    boost_asio_async_write_with_sentinels(
            *_socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(writeq, ec, bytes_transferred); });
#else
    write_channel_used = false;
    write_cumulative_transfer = 0;
    write_offset = 0;
    write_current_writeq = writeq;
    write_current_write = write;

    std::unique_lock<std::mutex> lock(write_thread_mutex);
    write_thread_trigger = true;
    lock.unlock();
    write_thread_trigger_cv.notify_one();

    write_next_compressed_chunk(writeq, write);
#endif
}

#ifdef IO_LINK_COMPRESSION
void DataStream::write_next_compressed_chunk(writeq_type *writeq, std::shared_ptr<DataSending> write) {
    std::unique_lock<std::mutex> lock(write_queue_mutex);
    if (write_channel_used) {
        // We're already inside a boost::asio::async_write call, so we can't initiate another one until it finishes
        return;
    }

    if ((write->size() - write_offset) >= CHUNK_SIZE) {
        if (write_queue.empty()) {
            // No compressed chunk is yet available
            return;
        }

        const auto &chunk = write_queue.front();
        write_channel_used = true;
        lock.unlock();

        // Chunk I/O
        std::array<boost::asio::const_buffer, 2> buffers = {
                boost::asio::buffer(&chunk.size, sizeof(size_t)),
                boost::asio::buffer(chunk.ptr, chunk.size),
        };
        boost_asio_async_write_with_sentinels(*_socket, buffers,
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             assert(!ec);

             std::unique_lock<std::mutex> lock(write_queue_mutex);
             write_channel_used = false;
             write_queue.pop();
             lock.unlock();

             write_cumulative_transfer += bytes_transferred;
             write_offset += CHUNK_SIZE;

             write_next_compressed_chunk(writeq, write);
         });
    } else {
        // Always write the last incomplete chunk of the input uncompressed
        write_channel_used = true;
        lock.unlock();

        boost_asio_async_write_with_sentinels(*_socket,
                boost::asio::buffer((uint8_t *) write->ptr() + write_offset, write->size() - write_offset),
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             assert(!ec);

             std::unique_lock<std::mutex> lock(write_queue_mutex);
             write_channel_used = false;
             lock.unlock();

             write_cumulative_transfer += bytes_transferred;
             write_offset = writeq->size();

             handle_write(writeq, boost::system::error_code(), write_cumulative_transfer);
         });
    }
}

void DataStream::start_compress_thread() {
    while (true) {
        std::unique_lock<std::mutex> lock(write_thread_mutex);
        while (!write_thread_trigger)
            write_thread_trigger_cv.wait(lock);
        write_thread_trigger = false;
        if (write_thread_trigger_quit)
            return;
        lock.unlock();

        auto writeq = write_current_writeq;
        auto write = write_current_write;

        for (size_t offset = 0; (write->size() - offset) >= CHUNK_SIZE; offset += CHUNK_SIZE) {
            if (write->skip_compress_step()) {
                auto is_compressed = std::equal((uint8_t *) write->ptr() + offset,
                                                (uint8_t *) write->ptr() + offset + sizeof(CHUNK_MAGIC), CHUNK_MAGIC);

                uint8_t *chunk_buffer = is_compressed
                                        ? (uint8_t *) write->ptr() + offset + sizeof(CHUNK_MAGIC) + sizeof(size_t)
                                        : (uint8_t *) write->ptr() + offset;
                size_t chunk_buffer_size = is_compressed ?
                                           *(size_t *) ((uint8_t *) write->ptr() + offset + sizeof(CHUNK_MAGIC))
                                                         : CHUNK_SIZE;

                write_chunk chunk(chunk_buffer, chunk_buffer_size, false);

                std::unique_lock<std::mutex> lock(write_queue_mutex);
                write_queue.push(std::move(chunk));
                lock.unlock();
            } else {
                // Compress chunk
                std::unique_ptr<uint8_t[]> compress_buffer(new uint8_t[2 * CHUNK_SIZE]);
                size_t compressed_size = 2 * CHUNK_SIZE;

                int ret = lib842_compress((uint8_t *) write->ptr() + offset, CHUNK_SIZE, compress_buffer.get(),
                                          &compressed_size);
                assert(ret == 0);

                // Push into the chunk queue
                auto compressible = compressed_size <= COMPRESSIBLE_THRESHOLD;
                uint8_t *chunk_buffer = compressible ? compress_buffer.release() : (uint8_t *) write->ptr() + offset;
                size_t chunk_buffer_size = compressible ? compressed_size : CHUNK_SIZE;

                write_chunk chunk(chunk_buffer, chunk_buffer_size, compressible);

                std::unique_lock<std::mutex> lock(write_queue_mutex);
                write_queue.push(std::move(chunk));
                lock.unlock();
            }

            write_next_compressed_chunk(writeq, write);
        }
    }
}
#endif

void DataStream::handle_write(
        writeq_type *writeq,
        const boost::system::error_code& ec,
        size_t bytes_transferred) {
    // current write is first element in writeq, so writeq must be non-empty
    assert(writeq /* ouch! */ && !writeq->empty());
    writeq->front()->onFinish(ec, bytes_transferred);
    writeq->pop();

    if (ec) {
        // TODO Handle errors
    }

    start_write(writeq); // process remaining writes
}

} // namespace comm

} // namespace dclasio
