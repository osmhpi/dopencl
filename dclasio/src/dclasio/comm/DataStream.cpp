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
#include <dcl/DCLTypes.h>

#include <boost/log/trivial.hpp>

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

// TODOXXX: This class has grown way too large and has way too many responsabilities
// It should be split into multiple files/classes like:
// * Receiving DataStream
// * Sending DataStream
// * Decompression pool
// * Compression pool
// * Boost.Asio sentinel helper

#ifdef IO_LINK_COMPRESSION

// Those constants must be synchronized with the constants in lib842 (cl842)
// for the integration with OpenCL-based decompression to work
#include <cl842.hpp>
#define CHUNK_SIZE CL842_CHUNK_SIZE
#define COMPRESSED_CHUNK_MAGIC CL842_COMPRESSED_CHUNK_MAGIC
#define COMPRESSIBLE_THRESHOLD ((CHUNK_SIZE - sizeof(COMPRESSED_CHUNK_MAGIC) - sizeof(uint64_t)))

// Configuration for the number of threads to use for compression or decompression
// If the value is 0, the hardware concurrency level (~= number of logical cores) is used
static constexpr unsigned int DEFAULT_NUM_COMPRESS_THREADS = 0;
static constexpr unsigned int DEFAULT_NUM_DECOMPRESS_THREADS = 0;

static unsigned int determine_num_threads(unsigned int preferred_num_threads) {
    if (preferred_num_threads != 0)
        return preferred_num_threads;

    static unsigned int hardware_concurrency = std::thread::hardware_concurrency();
    if (hardware_concurrency == 0) {
        BOOST_LOG_TRIVIAL(warning) << __func__ << ": "
            << "std::thread::hardware_concurrency() returned 0, using 1 thread"
            << std::endl;
        return 1;
    }

    return hardware_concurrency;
}

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
        _socket(socket), _receiving(false), _sending(false),

        _compress_threads(determine_num_threads(DEFAULT_NUM_COMPRESS_THREADS)),
        _compress_trigger(false), _compress_quit(false),
        _compress_start_barrier(_compress_threads.size()),
        _compress_finish_barrier(_compress_threads.size()+1),

        _decompress_threads(determine_num_threads(DEFAULT_NUM_DECOMPRESS_THREADS)),
        _decompress_finish_barrier(_decompress_threads.size()) {
    // TODO Ensure that socket is connected
    _remote_endpoint = _socket->remote_endpoint();

#ifdef IO_LINK_COMPRESSION
    start_decompress_threads();
    start_compress_threads();
#endif
}

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket,
        boost::asio::ip::tcp::endpoint remote_endpoint) :
        _socket(socket), _remote_endpoint(remote_endpoint), _receiving(false), _sending(false),

        _compress_threads(determine_num_threads(DEFAULT_NUM_COMPRESS_THREADS)),
        _compress_trigger(false), _compress_quit(false),
        _compress_start_barrier(_compress_threads.size()),
        _compress_finish_barrier(_compress_threads.size()+1),

        _decompress_threads(determine_num_threads(DEFAULT_NUM_DECOMPRESS_THREADS)),
        _decompress_finish_barrier(_decompress_threads.size()) {
    assert(!socket->is_open()); // socket must not be connect

#ifdef IO_LINK_COMPRESSION
    start_decompress_threads();
    start_compress_threads();
#endif
}

DataStream::~DataStream() {
#ifdef IO_LINK_COMPRESSION
    {
        std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
        _decompress_queue.push(decompress_message_quit());
        _decompress_queue_available.notify_all();
    }
    for (auto &t : _decompress_threads)
        t.join();

    {
        std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
        _compress_trigger = true;
        _compress_quit = true;
        _compress_trigger_changed.notify_all();
    }
    for (auto &t : _compress_threads)
        t.join();
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
    BOOST_LOG_TRIVIAL(trace)
            << "Sent process identification message for data stream (pid=" << pid << ')'
            << std::endl;

#if USE_DATA_STREAM_RESPONSE
    // receive response
    buf.resize(sizeof(dcl::process_id));
    boost::asio::read(*_socket, boost::asio::buffer(buf.data(), buf.size()));
    buf >> pid;
    BOOST_LOG_TRIVIAL(trace)
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
        size_t size, void *ptr, bool skip_compress_step, const std::shared_ptr<dcl::Completable> &trigger_event) {
    // START UBER HACK
    static constexpr size_t SUPERBLOCK_MAX_SIZE = static_cast<size_t>(1) << 29; // 512 MiB
    if (size > SUPERBLOCK_MAX_SIZE) {
        size_t num_superblocks = (size + SUPERBLOCK_MAX_SIZE - 1) / SUPERBLOCK_MAX_SIZE;
        std::shared_ptr<DataReceipt> callback;

        for (size_t i = 0; i < num_superblocks; i++) {
            size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
            size_t superblock_size = std::min(size - superblock_offset, SUPERBLOCK_MAX_SIZE);
            callback = read(superblock_size, static_cast<uint8_t *>(ptr) + superblock_offset, skip_compress_step, trigger_event);
        }

        return callback;
    }
    // END UBER HACK

    auto read(std::make_shared<DataReceipt>(size, ptr, skip_compress_step, trigger_event));

    std::unique_lock<std::mutex> lock(_readq_mtx);
    if ((_receiving)) {
        _readq.push(read);
    } else {
        // start read loop
        _receiving = true;
        lock.unlock();

        schedule_read(new readq_type({ read }));
    }

    return read;
}


std::shared_ptr<DataSending> DataStream::write(
        size_t size, const void *ptr, bool skip_compress_step, const std::shared_ptr<dcl::Completable> &trigger_event) {
    // START UBER HACK
    static constexpr size_t SUPERBLOCK_MAX_SIZE = static_cast<size_t>(1) << 29; // 512 MiB
    if (size > SUPERBLOCK_MAX_SIZE) {
        size_t num_superblocks = (size + SUPERBLOCK_MAX_SIZE - 1) / SUPERBLOCK_MAX_SIZE;
        std::shared_ptr<DataSending> callback;

        for (size_t i = 0; i < num_superblocks; i++) {
            size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
            size_t superblock_size = std::min(size - superblock_offset, SUPERBLOCK_MAX_SIZE);
            callback = write(superblock_size, static_cast<const uint8_t *>(ptr) + superblock_offset, skip_compress_step, trigger_event);
        }

        return callback;
    }
    // END UBER HACK

    auto write(std::make_shared<DataSending>(size, ptr, skip_compress_step, trigger_event));

    std::unique_lock<std::mutex> lock(_writeq_mtx);
    if (_sending) {
        _writeq.push(write);
    } else {
        // start write loop
        _sending = true;
        lock.unlock();

        schedule_write(new writeq_type({ write }));
    }

    return write;
}

void DataStream::schedule_read(readq_type *readq) {
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
    if (read->trigger_event() != nullptr) {
        // If a event to wait was given, start the read after the event callback
        // TODOXXX: Handle the case where the event to wait returns an error
        // (the lines deleted when this comment was introduced could serve as a reference)
        read->trigger_event()->setCallback([this, readq](cl_int status) {
            assert(status == CL_SUCCESS); // TODOXXX Handle errors
            start_read(readq);
        });
    } else {
        // If no event to wait was given, start the read immediately
        start_read(readq);
    }
}

void DataStream::start_read(readq_type *readq) {
    auto& read = readq->front();
    read->onStart();
#ifndef IO_LINK_COMPRESSION
    boost_asio_async_read_with_sentinels(
                *_socket, boost::asio::buffer(read->ptr(), read->size()),
                [this, readq](const boost::system::error_code& ec, size_t bytes_transferred){
                        handle_read(readq, ec, bytes_transferred); });
#else
    _decompress_working_thread_count = 0;
    _read_io_total_bytes_transferred = 0;
    _read_io_num_chunks_remaining = read->size() / CHUNK_SIZE;

    read_next_compressed_chunk(readq, read);
#endif
}

#ifdef IO_LINK_COMPRESSION
void DataStream::read_next_compressed_chunk(readq_type *readq, std::shared_ptr<DataReceipt> read) {
    if (_read_io_num_chunks_remaining > 0) {
        // Read header containing the offset and size of the current chunk
        std::array<boost::asio::mutable_buffer, 2> buffers = {
                boost::asio::buffer(&_read_io_destination_offset, sizeof(size_t)),
                boost::asio::buffer(&_read_io_buffer_size, sizeof(size_t))
        };
        boost_asio_async_read_with_sentinels(*_socket, buffers,
                [this, readq, read] (const boost::system::error_code& ec, size_t bytes_transferred){
            assert(!ec);
            assert(_read_io_destination_offset <= read->size() - CHUNK_SIZE);
            _read_io_total_bytes_transferred += bytes_transferred;

            if (_read_io_buffer_size <= COMPRESSIBLE_THRESHOLD && !read->skip_compress_step()) {
                // Read the compressed chunk into a secondary buffer to be decompressed later
                _read_io_buffer.resize(_read_io_buffer_size);
                boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer(_read_io_buffer.data(), _read_io_buffer_size),
                    [this, readq, read] (const boost::system::error_code& ec, size_t bytes_transferred) {
                    assert(!ec);
                    _read_io_total_bytes_transferred += bytes_transferred;

                    // Push into the queue for decompression
                    decompress_message_decompress dm = {
                            .compressed_data = std::move(_read_io_buffer),
                            .destination = static_cast<uint8_t *>(read->ptr()) + _read_io_destination_offset
                    };
                    decompress_message_t msg(std::move(dm));

                    {
                        std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
                        _decompress_queue.push(std::move(msg));
                        _decompress_queue_available.notify_one();
                    }

                    _read_io_num_chunks_remaining--;

                    read_next_compressed_chunk(readq, read);
                });
            } else {
                // Read the chunk directly in its final destination in the destination buffer
                uint8_t *destination = static_cast<uint8_t *>(read->ptr()) + _read_io_destination_offset;

                if (_read_io_buffer_size <= COMPRESSIBLE_THRESHOLD && read->skip_compress_step()) {
                    std::copy(COMPRESSED_CHUNK_MAGIC, COMPRESSED_CHUNK_MAGIC + sizeof(COMPRESSED_CHUNK_MAGIC), destination);
                    *reinterpret_cast<uint64_t *>((destination + sizeof(COMPRESSED_CHUNK_MAGIC))) = _read_io_buffer_size;
                    destination += CHUNK_SIZE - _read_io_buffer_size; // Write compressed data at the end
                } else {
                    assert(_read_io_buffer_size == CHUNK_SIZE); // Chunk is read uncompressed
                }
                boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer(destination, _read_io_buffer_size),
                    [this, readq, read] (const boost::system::error_code& ec, size_t bytes_transferred) {
                    assert(!ec);
                    _read_io_total_bytes_transferred += bytes_transferred;
                    _read_io_num_chunks_remaining--;

                    read_next_compressed_chunk(readq, read);
                });
            }
        });
    } else {
        // Always read the last incomplete chunk of the input uncompressed
        auto last_chunk_destination_ptr =  static_cast<uint8_t *>(read->ptr()) + (read->size() & ~(CHUNK_SIZE - 1));
        auto last_chunk_size = read->size() & (CHUNK_SIZE - 1);

        boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer(last_chunk_destination_ptr, last_chunk_size),
                                [this, readq, read](const boost::system::error_code &ec, size_t bytes_transferred) {
            assert(!ec);
            _read_io_total_bytes_transferred += bytes_transferred;

            std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
            bool done = _decompress_working_thread_count == 0 && _decompress_queue.empty();
            // If there are still decompression operations active, we need to wait
            // until they finish to finalize the entire operation
            // In this case, transfer the responsability of finalizing to the decompression threads
            if (!done) {
                _decompress_queue.push(decompress_message_finalize{.readq = readq});
                _decompress_queue_available.notify_all();
            }
            lock.unlock();

            // Otherwise, if all decompression threads finished as well,
            // finalize the entire operation as soon as possible here
            if (done) {
                handle_read(readq, boost::system::error_code(), _read_io_total_bytes_transferred);
            }
        });
    }
}

void DataStream::start_decompress_threads() {
    for (size_t i = 0; i < _decompress_threads.size(); i++) {
        _decompress_threads[i] = std::thread{&DataStream::loop_decompress_thread, this, i};
    }
}

void DataStream::loop_decompress_thread(size_t thread_id) {
    static constexpr int CHUNKVARIANT_WHICH_MESSAGE_DECOMPRESS = 0;
    static constexpr int CHUNKVARIANT_WHICH_MESSAGE_FINALIZE = 1;
    static constexpr int CHUNKVARIANT_WHICH_MESSAGE_QUIT = 2;

    while (true) {
        // (Blocking) pop from the chunk queue
        std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
        _decompress_queue_available.wait(lock, [this] { return !_decompress_queue.empty(); });
        if (_decompress_queue.front().which() == CHUNKVARIANT_WHICH_MESSAGE_FINALIZE) {
            lock.unlock();

            // Wait until all threads have got the "finalize" message
            _decompress_finish_barrier.wait();

            // "Leader" thread finalizes the write and pops the message from the queue
            if (thread_id == 0) {
                auto finalize_message = boost::get<decompress_message_finalize>(_decompress_queue.front());
                _decompress_queue.pop();
                handle_read(finalize_message.readq, boost::system::error_code(), _read_io_total_bytes_transferred);
            }

            // Once write is finalized, wait again
            _decompress_finish_barrier.wait();
        } else if (_decompress_queue.front().which() == CHUNKVARIANT_WHICH_MESSAGE_QUIT) {
            break;
        } else if (_decompress_queue.front().which() == CHUNKVARIANT_WHICH_MESSAGE_DECOMPRESS) {
            auto chunkVariant = std::move(_decompress_queue.front());
            _decompress_queue.pop();
            _decompress_working_thread_count++;

#if 0
            // Tentative code for GPU-based decompression
            // GPU-based decompression relies on processing multiple chunks in parallel,
            // so we need to gather a few chunks here and then decompress an entire batch at once
            static constexpr size_t NUM_PARALLEL_GPU_DECOMPRESS_CHUNKS = 16;
            std::array<decompress_message_decompress, NUM_PARALLEL_GPU_DECOMPRESS_CHUNKS> chunk_batch;
            size_t num_chunks = 1;
            chunk_batch[0] = std::move(boost::get<decompress_message_decompress>(chunkVariant));

            while (num_chunks < chunk_batch.size()) {
                _decompress_queue_available.wait(lock, [this] { return !_decompress_queue.empty(); });
                if (_decompress_queue.front().which() != CHUNKVARIANT_WHICH_MESSAGE_DECOMPRESS) {
                    // All read IO operations already finished, and wants us to quit or finalize,
                    // so finish this batch and then do it
                    // Possible optimization: Make the read send a otherwise useless "pre-finish"
                    // message when the last (uncompressed) read begins so we can start earlier here
                    break;
                }
                chunk_batch[num_chunks++] = std::move(boost::get<decompress_message_decompress>(_decompress_queue.front()));
                _decompress_queue.pop();
            }
            lock.unlock();

            // TODOXXX: Uncompress the chunks on the GPU
            // To be determined: Which command queue to use? I believe we need to create a new one here
            // just for decompression
            for (size_t i = 0; i < num_chunks; i++) {
                assert(chunk_batch[i].compressed_data.size() <= COMPRESSIBLE_THRESHOLD);

                size_t uncompressed_size = CHUNK_SIZE;
                int ret = lib842_decompress(chunk_batch[i].compressed_data.data(),
                                            chunk_batch[i].compressed_data.size(),
                                            static_cast<uint8_t *>(chunk_batch[i].destination), &uncompressed_size);
                assert(ret == 0);
                assert(uncompressed_size == CHUNK_SIZE);
            }
#else
            lock.unlock();
            auto chunk = std::move(boost::get<decompress_message_decompress>(chunkVariant));
            auto destination = static_cast<uint8_t *>(chunk.destination);

            assert(chunk.compressed_data.size() <= COMPRESSIBLE_THRESHOLD);

            size_t uncompressed_size = CHUNK_SIZE;
            int ret = lib842_decompress(chunk.compressed_data.data(),
                                        chunk.compressed_data.size(),
                                        destination, &uncompressed_size);
            assert(ret == 0);
            assert(uncompressed_size == CHUNK_SIZE);
#endif

            lock.lock();
            _decompress_working_thread_count--;
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

    schedule_read(readq); // process remaining reads
}

void DataStream::schedule_write(writeq_type *writeq) {
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
    if (write->trigger_event() != nullptr) {
        // If a event to wait was given, start the write after the event callback
        // TODOXXX: Handle the case where the event to wait returns an error
        // (the lines deleted when this comment was introduced could serve as a reference)
        write->trigger_event()->setCallback([this, writeq](cl_int status) {
            assert(status == CL_SUCCESS); // TODOXXX Handle errors
            start_write(writeq);
        });
    } else {
        // If a event to wait was given, start the write immediately
        start_write(writeq);
    }
}

void DataStream::start_write(writeq_type *writeq) {
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
    _write_io_channel_busy = false;
    _write_io_total_bytes_transferred = 0;
    _write_io_num_chunks_remaining = write->size() / CHUNK_SIZE;

    if (write->size() >= CHUNK_SIZE) {
        _compress_current_writeq = writeq;
        _compress_current_write = write;
        _compress_current_offset = 0;

        std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
        _compress_trigger = true;
        _compress_trigger_changed.notify_all();
    }

    try_write_next_compressed_chunk(writeq, write);
#endif
}

#ifdef IO_LINK_COMPRESSION
void DataStream::try_write_next_compressed_chunk(writeq_type *writeq, std::shared_ptr<DataSending> write) {
    std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
    if (_write_io_channel_busy) {
        // We're already inside a boost::asio::async_write call, so we can't initiate another one until it finishes
        return;
    }

    if (_write_io_num_chunks_remaining > 0) {
        if (_write_io_queue.empty()) {
            // No compressed chunk is yet available
            return;
        }

        const auto &chunk = _write_io_queue.front();
        _write_io_channel_busy = true;
        lock.unlock();

        // Chunk I/O
        std::array<boost::asio::const_buffer, 3> buffers = {
                boost::asio::buffer(&chunk.source_offset, sizeof(size_t)),
                boost::asio::buffer(&chunk.size, sizeof(size_t)),
                boost::asio::buffer(chunk.data.get(), chunk.size),
        };
        boost_asio_async_write_with_sentinels(*_socket, buffers,
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             assert(!ec);

             std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
             _write_io_channel_busy = false;
             _write_io_queue.pop();
             lock.unlock();

             _write_io_total_bytes_transferred += bytes_transferred;
             _write_io_num_chunks_remaining--;

             try_write_next_compressed_chunk(writeq, write);
         });
    } else {
        // Always write the last incomplete chunk of the input uncompressed
        _write_io_channel_busy = true;
        lock.unlock();

        auto last_chunk_source_ptr =  static_cast<const uint8_t *>(write->ptr()) + (write->size() & ~(CHUNK_SIZE - 1));
        auto last_chunk_size = write->size() & (CHUNK_SIZE - 1);

        boost_asio_async_write_with_sentinels(*_socket,
                boost::asio::buffer(last_chunk_source_ptr, last_chunk_size),
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             assert(!ec);

             std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
             _write_io_channel_busy = false;
             lock.unlock();

             _write_io_total_bytes_transferred += bytes_transferred;

             // The data transfer thread also joins the final barrier for the compression
             // threads before finishing the write, to ensure resources are not released
             // while a compression thread still hasn't realized all work is finished
             if (write->size() >= CHUNK_SIZE) {
                _compress_finish_barrier.wait();
             }

             handle_write(writeq, boost::system::error_code(), _write_io_total_bytes_transferred);
         });
    }
}

void DataStream::start_compress_threads() {
    for (size_t i = 0; i < _compress_threads.size(); i++) {
        _compress_threads[i] = std::thread{&DataStream::loop_compress_thread, this, i};
    }
}

void DataStream::loop_compress_thread(size_t thread_id) {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
            _compress_trigger_changed.wait(lock, [this] { return _compress_trigger; });
            if (_compress_quit)
                return;
        }

        _compress_start_barrier.wait();
        _compress_trigger = false;

        auto writeq = _compress_current_writeq;
        auto write = _compress_current_write;
        auto last_valid_offset = write->size() & ~(CHUNK_SIZE-1);

        while (true) {
            size_t offset = _compress_current_offset.fetch_add(CHUNK_SIZE);
            if (offset >= last_valid_offset) {
                break;
            }

            auto source = static_cast<const uint8_t *>(write->ptr()) + offset;

            if (write->skip_compress_step()) {
                auto is_compressed = std::equal(source,source + sizeof(COMPRESSED_CHUNK_MAGIC), COMPRESSED_CHUNK_MAGIC);

                auto chunk_buffer_size = is_compressed
                                         ? *reinterpret_cast<const uint64_t *>((source + sizeof(COMPRESSED_CHUNK_MAGIC)))
                                         : CHUNK_SIZE;
                auto chunk_buffer = is_compressed
                        ? source + CHUNK_SIZE - chunk_buffer_size
                        : source;

                write_chunk chunk {
                    .data = std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter>(
                            chunk_buffer, ConditionalOwnerDeleter(false)),
                    .source_offset = offset,
                    .size = chunk_buffer_size
                };

                std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
                _write_io_queue.push(std::move(chunk));
            } else {
                // Compress chunk
                std::unique_ptr<uint8_t[]> compress_buffer(new uint8_t[2 * CHUNK_SIZE]);
                size_t compressed_size = 2 * CHUNK_SIZE;

                int ret = lib842_compress(source, CHUNK_SIZE, compress_buffer.get(),
                                          &compressed_size);
                assert(ret == 0);

                // Push into the chunk queue
                auto compressible = compressed_size <= COMPRESSIBLE_THRESHOLD;

                // If the chunk is compressible, transfer ownership of the compressed buffer,
                // otherwise use the uncompressed buffer and destroy the compressed buffer
                write_chunk chunk {
                    .data = std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter>(
                            compressible ? compress_buffer.release() : source,
                            ConditionalOwnerDeleter(compressible)),
                    .source_offset = offset,
                    .size = compressible ? compressed_size : CHUNK_SIZE
                };

                std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
                _write_io_queue.push(std::move(chunk));
            }

            try_write_next_compressed_chunk(writeq, write);
        }

        _compress_finish_barrier.wait();
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

    schedule_write(writeq); // process remaining writes
}

} // namespace comm

} // namespace dclasio
