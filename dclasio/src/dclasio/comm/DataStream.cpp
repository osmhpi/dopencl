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
#include <thread>
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

namespace dclasio {

namespace comm {

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket) :
        _socket(socket), _receiving(false), _sending(false) {
    // TODO Ensure that socket is connected
    _remote_endpoint = _socket->remote_endpoint();
}

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket,
        boost::asio::ip::tcp::endpoint remote_endpoint) :
        _socket(socket), _remote_endpoint(remote_endpoint), _receiving(false), _sending(false) {
    assert(!socket->is_open()); // socket must not be connect
}

DataStream::~DataStream() {
}

dcl::process_id DataStream::connect(
        dcl::process_id pid) {
    _socket->connect(_remote_endpoint); // connect socket to remote endpoint

    // send process ID to remote process via data stream
    // TODO Encode local process type and data stream protocol
    dcl::ByteBuffer buf;
    buf << pid << uint8_t(0) << uint8_t(0);
    boost::asio::write(*_socket, boost::asio::buffer(buf.begin(), buf.size()));
    dcl::util::Logger << dcl::util::Verbose
            << "Sent process identification message for data stream (pid=" << pid << ')'
            << std::endl;

#if USE_DATA_STREAM_RESPONSE
    // receive response
    buf.resize(sizeof(dcl::process_id));
    boost::asio::read(*_socket, boost::asio::buffer(buf.begin(), buf.size()));
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
        size_t size, void *ptr) {
    auto read(std::make_shared<DataReceipt>(size, ptr));

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
        size_t size, const void *ptr) {
    auto write(std::make_shared<DataSending>(size, ptr));

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
    boost::asio::async_read(
            *_socket, boost::asio::buffer(read->ptr(), read->size()),
            [this, readq](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_read(readq, ec, bytes_transferred); });
#else
    auto queue = new std::queue<std::vector<uint8_t>>();
    auto queue_mtx = new std::mutex();
    auto queue_available = new std::condition_variable();

    // TODOXXX: Use faster (GPU, HW) compression
    // TODOXXX: Does it make sense to spawn multiple compression (producer) threads?
    //          I guess not due to the GPU/HW methods already being parallel themselves
    // TODOXXX: Does it make sense to use asynchronous I/O here (can this thread be avoided?)

    auto io_thread = new std::thread([this, readq, read, queue, queue_mtx, queue_available] {
        size_t remaining_offset = 0, remaining_size = read->size();

        while (remaining_size >= CHUNK_SIZE) {
            size_t compressed_size;
            boost::asio::read(*_socket, boost::asio::buffer(&compressed_size, sizeof(size_t)));

            std::vector<uint8_t> compress_buffer(compressed_size);
            boost::asio::read(*_socket, boost::asio::buffer(compress_buffer.data(), compressed_size));

            // Push into the chunk queue
            std::unique_lock<std::mutex> lock(*queue_mtx);
            queue->push(std::move(compress_buffer));
            lock.unlock();
            queue_available->notify_one();

            remaining_offset += CHUNK_SIZE;
            remaining_size -= CHUNK_SIZE;
        }
    });

    auto compression_thread = new std::thread([this, readq, read, queue, queue_mtx, queue_available] {
        size_t remaining_offset = 0, remaining_size = read->size();

        while (remaining_size >= CHUNK_SIZE) {
            // (Blocking) pop from the chunk queue
            std::unique_lock<std::mutex> lock(*queue_mtx);
            while (queue->empty())
                queue_available->wait(lock);
            auto chunk = std::move(queue->front());
            queue->pop();
            lock.unlock();

            // Decompression
            size_t uncompressed_size = CHUNK_SIZE;
            int ret = lib842_decompress((uint8_t *)chunk.data(), chunk.size(),
                    (uint8_t *)read->ptr() + remaining_offset, &uncompressed_size);
            assert(ret == 0);
            assert(uncompressed_size == CHUNK_SIZE);

            remaining_offset += CHUNK_SIZE;
            remaining_size -= CHUNK_SIZE;
        }

        // Always read the last incomplete chunk of the input uncompressed
        boost::asio::read(*_socket,
                boost::asio::buffer((uint8_t *)read->ptr() + remaining_offset, remaining_size));
        handle_read(readq, boost::system::error_code(), read->size());
    });

    // TODOXXX: Release all resources allocated there! (Memory leaks galore)
#endif
}

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

struct write_chunk {
    uint8_t *ptr;
    size_t size;
    bool is_owner;

    write_chunk(uint8_t *ptr, size_t size, bool is_owner)
        : ptr(ptr), size(size), is_owner(is_owner) {
    }

    ~write_chunk() {
        if (is_owner) {
            delete[] ptr;
        }
    }

    write_chunk(const write_chunk& other) = delete;
    write_chunk& operator=(const write_chunk& other) = delete;

    write_chunk(write_chunk&& other) noexcept // move constructor
        : ptr(other.ptr), size(other.size), is_owner(other.is_owner) {
        other.ptr = nullptr; // Avoid delete
    }

    write_chunk& operator=(write_chunk&& other) noexcept {
        ptr = other.ptr;
        size = other.size;
        is_owner = other.is_owner;
        other.ptr = nullptr; // Avoid delete
        return *this;
    }
};

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
    boost::asio::async_write(
            *_socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq{std::move(writeq)}](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(std::move(writeq), ec, bytes_transferred); });
     */
#ifndef IO_LINK_COMPRESSION
    boost::asio::async_write(
            *_socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(writeq, ec, bytes_transferred); });
#else
    auto queue = new std::queue<write_chunk>();
    auto queue_mtx = new std::mutex();
    auto queue_available = new std::condition_variable();

    // TODOXXX: Use faster (GPU, HW) compression
    // TODOXXX: Does it make sense to spawn multiple compression (producer) threads?
    //          I guess not due to the GPU/HW methods already being parallel themselves
    // TODOXXX: Does it make sense to use asynchronous I/O here (can this thread be avoided?)

    auto compression_thread = new std::thread([this, writeq, write, queue, queue_mtx, queue_available] {
        size_t remaining_offset = 0, remaining_size = write->size();
        while (remaining_size >= CHUNK_SIZE) {
            // Compress chunk
            auto compress_buffer = new uint8_t[2 * CHUNK_SIZE];
            size_t compressed_size = 2 * CHUNK_SIZE;

            int ret = lib842_compress((uint8_t *)write->ptr() + remaining_offset, CHUNK_SIZE, compress_buffer,
                                     &compressed_size);
            assert(ret == 0);

            // Push into the chunk queue
            write_chunk chunk(compress_buffer, compressed_size, true);

            std::unique_lock<std::mutex> lock(*queue_mtx);
            queue->push(std::move(chunk));
            lock.unlock();
            queue_available->notify_one();

            remaining_offset += CHUNK_SIZE;
            remaining_size -= CHUNK_SIZE;
        }
    });

    auto io_thread = new std::thread([this, writeq, write, queue, queue_mtx, queue_available] {
        size_t remaining_offset = 0, remaining_size = write->size();
        while (remaining_size >= CHUNK_SIZE) {
            // (Blocking) pop from the chunk queue
            std::unique_lock<std::mutex> lock(*queue_mtx);
            while (queue->empty())
                queue_available->wait(lock);
            auto chunk = std::move(queue->front());
            queue->pop();
            lock.unlock();

            // (Synchronous) chunk I/O
            std::array<boost::asio::const_buffer, 2> header_and_data = {
                    boost::asio::buffer(&chunk.size, sizeof(size_t)),
                    boost::asio::buffer(chunk.ptr, chunk.size)};
            boost::asio::write(*_socket, header_and_data);

            remaining_offset += CHUNK_SIZE;
            remaining_size -= CHUNK_SIZE;
        }

        // Always write the last incomplete chunk of the input uncompressed
        boost::asio::write(*_socket,
                boost::asio::buffer((uint8_t *)write->ptr() + remaining_offset, remaining_size));
        handle_write(std::move(writeq), boost::system::error_code(), write->size());
    });

    // TODOXXX: Release all resources allocated there! (Memory leaks galore)
#endif
}

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
