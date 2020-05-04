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
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <utility>

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
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

// TODOXXX: This class has grown way too large and has way too many responsabilities
// It should be split into multiple files/classes like:
// * Receiving DataStream
// * Sending DataStream
// * Decompression pool
// * Compression pool
// * Boost.Asio sentinel helper

#ifdef IO_LINK_COMPRESSION

static unsigned int determine_num_threads(const char *env_name) {
    // Configuration for the number of threads to use for compression or decompression
    const char *env_value = std::getenv(env_name);
    if (env_value != nullptr && std::atoi(env_value) > 0) {
        return (unsigned int)std::atoi(env_value);
    }

    // If the value is not specified (or invalid),
    // the hardware concurrency level (~= number of logical cores) is used
    static unsigned int hardware_concurrency = std::thread::hardware_concurrency();
    if (hardware_concurrency == 0) {
        dcl::util::Logger << dcl::util::Warning << __func__ << ": "
            << "std::thread::hardware_concurrency() returned 0, using 1 thread"
            << std::endl;
        return 1;
    }

    return hardware_concurrency;
}

#include <sw842.h>
#ifdef USE_HW_IO_LINK_COMPRESSION
// TODOXXX: Should add the code to spread the threads among NUMA zones? (From lib842 sample)
#include <hw842.h>
#endif

#if defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
static bool is_hw_io_link_compression_enabled() {
    static bool enabled = std::getenv("DCL_DISABLE_HW_IO_LINK_COMPRESSION") == nullptr;
    return enabled;
}
#endif

static int lib842_compress(const uint8_t *in, size_t ilen,
                           uint8_t *out, size_t *olen) {
#if defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
    if (is_hw_io_link_compression_enabled())
        return hw842_compress(in ,ilen, out, olen);
#endif

    return optsw842_compress(in, ilen, out, olen);
}

static int lib842_decompress(const uint8_t *in, size_t ilen,
                             uint8_t *out, size_t *olen) {
#if defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
    if (is_hw_io_link_compression_enabled())
        return hw842_decompress(in ,ilen, out, olen);
#endif

    return optsw842_decompress(in, ilen, out, olen);
}

// TODOXXX: This is a variation of the above but used to get some hacky code to work. Remove me.
static void next_transfer_id_uberhax(dcl::transfer_id &transfer_id) {
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
        _socket(socket),
        _read_state(receiving_state::idle), _sending(false)
#ifdef IO_LINK_COMPRESSION
        , _compress_threads(determine_num_threads("DCL_IO_LINK_NUM_COMPRESS_THREADS")),
        _compress_trigger(false), _compress_quit(false),
        _compress_start_barrier(_compress_threads.size()),
        _compress_finish_barrier(_compress_threads.size()+1),

        _decompress_threads(determine_num_threads("DCL_IO_LINK_NUM_DECOMPRESS_THREADS")),
        _decompress_finish_barrier(_decompress_threads.size())
#endif
{
    // TODO Ensure that socket is connected
    _remote_endpoint = _socket->remote_endpoint();

#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        start_decompress_threads();
        start_compress_threads();
    }
#endif
}

DataStream::DataStream(
        const std::shared_ptr<boost::asio::ip::tcp::socket>& socket,
        boost::asio::ip::tcp::endpoint remote_endpoint) :
        _socket(socket), _remote_endpoint(remote_endpoint),
        _read_state(receiving_state::idle), _sending(false)
#ifdef IO_LINK_COMPRESSION
        , _compress_threads(determine_num_threads("DCL_IO_LINK_NUM_COMPRESS_THREADS")),
        _compress_trigger(false), _compress_quit(false),
        _compress_start_barrier(_compress_threads.size()),
        _compress_finish_barrier(_compress_threads.size()+1),

        _decompress_threads(determine_num_threads("DCL_IO_LINK_NUM_DECOMPRESS_THREADS")),
        _decompress_finish_barrier(_decompress_threads.size())
#endif
{
    assert(!socket->is_open()); // socket must not be connect

#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        start_decompress_threads();
        start_compress_threads();
    }
#endif
}

DataStream::~DataStream() {
#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
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
    }
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

std::shared_ptr<DataReceipt> DataStream::read(
        dcl::transfer_id transfer_id,
        size_t size, void *ptr, bool skip_compress_step,
        const std::shared_ptr<dcl::Completable> &trigger_event) {
#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        // START UBER HACK
        if (size > SUPERBLOCK_MAX_SIZE) {
            size_t num_superblocks = (size + SUPERBLOCK_MAX_SIZE - 1) / SUPERBLOCK_MAX_SIZE;
            std::shared_ptr<DataReceipt> callback;

            dcl::transfer_id split_transfer_id = transfer_id;
            for (size_t i = 0; i < num_superblocks; i++, next_transfer_id_uberhax(split_transfer_id)) {
                size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
                size_t superblock_size = std::min(size - superblock_offset, SUPERBLOCK_MAX_SIZE);
                callback = read(
                    split_transfer_id, superblock_size,
                    static_cast<uint8_t *>(ptr) + superblock_offset,
                    skip_compress_step, trigger_event);
            }

            return callback;
        }
        // END UBER HACK
    }
#endif

    auto read(std::make_shared<DataReceipt>(transfer_id, size, ptr, skip_compress_step));
    if (trigger_event != nullptr) {
        // If a event to wait was given, start the read after the event callback
        // TODOXXX: Handle the case where the event to wait returns an error
        trigger_event->setCallback([this, read](cl_int status) {
            assert(status == CL_SUCCESS); // TODOXXX Handle errors
            enqueue_read(read);
        });
    } else {
        // If no event to wait was given, start the read immediately
        enqueue_read(read);
    }
    return read;
}

void DataStream::enqueue_read(const std::shared_ptr<DataReceipt> &read) {
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

void DataStream::receive_matching_transfer_id() {
    boost_asio_async_read_with_sentinels(
        *_socket, boost::asio::buffer(_read_transfer_id.data, _read_transfer_id.size()),
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

void DataStream::start_read() {
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
        << "Start read of size " << _read_op->size()
        << std::endl;
#endif
    _read_op->onStart();

#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        _decompress_working_thread_count = 0;
        _read_io_total_bytes_transferred = 0;
        _read_io_num_blocks_remaining = _read_op->size() / NETWORK_BLOCK_SIZE;

        read_next_compressed_block();
        return;
    }
#endif

    boost_asio_async_read_with_sentinels(
                *_socket, boost::asio::buffer(_read_op->ptr(), _read_op->size()),
                [this](const boost::system::error_code& ec, size_t bytes_transferred){
                        handle_read(ec, bytes_transferred); });
}

#ifdef IO_LINK_COMPRESSION
void DataStream::read_next_compressed_block() {
    if (_read_io_num_blocks_remaining > 0) {
        // Read header containing the offset and size of the chunks in the current block
        std::array<boost::asio::mutable_buffer, 2> buffers = {
                boost::asio::buffer(&_read_io_destination_offset, sizeof(size_t)),
                boost::asio::buffer(_read_io_buffer_sizes.data(), NUM_CHUNKS_PER_NETWORK_BLOCK * sizeof(size_t))
        };
        boost_asio_async_read_with_sentinels(*_socket, buffers,
                [this] (const boost::system::error_code& ec, size_t bytes_transferred){
            assert(!ec);
            assert(_read_io_destination_offset <= _read_op->size() - NETWORK_BLOCK_SIZE);
            _read_io_total_bytes_transferred += bytes_transferred;

            std::array<boost::asio::mutable_buffer, NUM_CHUNKS_PER_NETWORK_BLOCK> recv_buffers;
            for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                assert(_read_io_buffer_sizes[i] > 0); // Chunk is read uncompressed
                if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && !_read_op->skip_compress_step()) {
                    // Read the compressed chunk into a secondary buffer to be decompressed later
                    // TODOXXX Should probably avoid doing separate allocations for chunks in a network block
                    _read_io_buffers[i].resize(_read_io_buffer_sizes[i]);
                    recv_buffers[i] = boost::asio::buffer(_read_io_buffers[i].data(), _read_io_buffer_sizes[i]);
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

            boost_asio_async_read_with_sentinels(*_socket, recv_buffers,
                [this] (const boost::system::error_code& ec, size_t bytes_transferred) {
                assert(!ec);
                _read_io_total_bytes_transferred += bytes_transferred;

                // Push into the queue for decompression
                decompress_message_decompress_block dm;
                bool should_uncompress_any = false;
                for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                    if (_read_io_buffer_sizes[i] <= COMPRESSIBLE_THRESHOLD && !_read_op->skip_compress_step()) {
                        dm.chunks[i] = decompress_chunk{
                                .compressed_data = std::move(_read_io_buffers[i]),
                                .destination = static_cast<uint8_t *>(_read_op->ptr()) + _read_io_destination_offset + i * CHUNK_SIZE
                        };
                        should_uncompress_any = true;
                    } else {
                        dm.chunks[i] = decompress_chunk{
                                .compressed_data = std::vector<uint8_t>(),
                                .destination = nullptr
                        };
                    }
                }

                if (should_uncompress_any) {
                    decompress_message_t msg(std::move(dm));

                    std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
                    _decompress_queue.push(std::move(msg));
                    _decompress_queue_available.notify_one();
                }

                _read_io_num_blocks_remaining--;

                read_next_compressed_block();
            });
        });
    } else {
        // Always read the last incomplete block of the input uncompressed
        auto last_block_destination_ptr =  static_cast<uint8_t *>(_read_op->ptr()) + (_read_op->size() & ~(NETWORK_BLOCK_SIZE - 1));
        auto last_block_size = _read_op->size() & (NETWORK_BLOCK_SIZE - 1);

        boost_asio_async_read_with_sentinels(*_socket, boost::asio::buffer(last_block_destination_ptr, last_block_size),
                                [this](const boost::system::error_code &ec, size_t bytes_transferred) {
            assert(!ec);
            _read_io_total_bytes_transferred += bytes_transferred;

            std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
            bool done = _decompress_working_thread_count == 0 && _decompress_queue.empty();
            // If there are still decompression operations active, we need to wait
            // until they finish to finalize the entire operation
            // In this case, transfer the responsability of finalizing to the decompression threads
            if (!done) {
                _decompress_queue.push(decompress_message_finalize{});
                _decompress_queue_available.notify_all();
            }
            lock.unlock();

            // Otherwise, if all decompression threads finished as well,
            // finalize the entire operation as soon as possible here
            if (done) {
                handle_read(boost::system::error_code(), _read_io_total_bytes_transferred);
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
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
        << "Start decompression thread with id " << thread_id
        << std::endl;
    size_t stat_handled_blocks = 0;
#endif

    static constexpr int VARIANT_WHICH_MESSAGE_DECOMPRESS = 0;
    static constexpr int VARIANT_WHICH_MESSAGE_FINALIZE = 1;
    static constexpr int VARIANT_WHICH_MESSAGE_QUIT = 2;

    while (true) {
        // (Blocking) pop from the chunk queue
        std::unique_lock<std::mutex> lock(_decompress_queue_mutex);
        _decompress_queue_available.wait(lock, [this] { return !_decompress_queue.empty(); });
        if (_decompress_queue.front().which() == VARIANT_WHICH_MESSAGE_FINALIZE) {
            lock.unlock();

            // Wait until all threads have got the "finalize" message
            _decompress_finish_barrier.wait();

            // "Leader" thread finalizes the write and pops the message from the queue
            if (thread_id == 0) {
                auto finalize_message = boost::get<decompress_message_finalize>(_decompress_queue.front());
                _decompress_queue.pop();
                handle_read(boost::system::error_code(), _read_io_total_bytes_transferred);
            }

            // Once write is finalized, wait again
            _decompress_finish_barrier.wait();
        } else if (_decompress_queue.front().which() == VARIANT_WHICH_MESSAGE_QUIT) {
            break;
        } else if (_decompress_queue.front().which() == VARIANT_WHICH_MESSAGE_DECOMPRESS) {
            auto blockVariant = std::move(_decompress_queue.front());
            _decompress_queue.pop();
            _decompress_working_thread_count++;

            lock.unlock();
#ifdef INDEPTH_TRACE
            stat_handled_blocks++;
#endif
            auto block = std::move(boost::get<decompress_message_decompress_block>(blockVariant));
            for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                const auto &chunk = block.chunks[i];
                if (chunk.compressed_data.empty() && chunk.destination == nullptr) {
                    // Chunk was transferred uncompressed, nothing to do
                    continue;
                }


                auto destination = static_cast<uint8_t *>(chunk.destination);

                assert(chunk.compressed_data.size() > 0 &&
                       chunk.compressed_data.size() <= COMPRESSIBLE_THRESHOLD);

                size_t uncompressed_size = CHUNK_SIZE;
                int ret = lib842_decompress(chunk.compressed_data.data(),
                                            chunk.compressed_data.size(),
                                            destination, &uncompressed_size);
                assert(ret == 0);
                assert(uncompressed_size == CHUNK_SIZE);
            }

            lock.lock();
            _decompress_working_thread_count--;
            lock.unlock();
        } else {
            assert(0);
        }
    }
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
        << "End decompression thread with id " << thread_id << " (stat_handled_blocks=" << stat_handled_blocks << ")"
        << std::endl;
#endif
}
#endif

void DataStream::handle_read(
        const boost::system::error_code& ec,
        size_t bytes_transferred) {
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
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

void DataStream::readToClBuffer(
        dcl::transfer_id transferId,
        size_t size,
        const cl::Context &context,
        const CL842DeviceDecompressor *cl842DeviceDecompressor,
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

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE) && defined(LIB842_HAVE_OPENCL)
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

        dcl::transfer_id split_transfer_id = transferId;
        for (size_t i = 0; i < num_superblocks; i++, next_transfer_id_uberhax(split_transfer_id)) {
            size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
            size_t superblock_size = std::min(size - i * SUPERBLOCK_MAX_SIZE, SUPERBLOCK_MAX_SIZE);

            // schedule local data transfer
            cl::UserEvent receiveEvent(context);
            std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(mapEvents[i]));
            read(split_transfer_id, superblock_size, ptrs[i], true, mapDataCompletable)
                ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));


            /* Enqueue unmap buffer (implicit upload) */
            cl::vector<cl::Event> unmapWaitList = {receiveEvent};
            commandQueue.enqueueUnmapMemObject(buffer, ptrs[i], &unmapWaitList, &unmapEvents[i]);
            // Rounds down (partial chunks are not compressed by DataStream)
            size_t chunksSize = superblock_size & ~(dcl::DataTransfer::COMPR842_CHUNK_SIZE - 1);
            if (chunksSize > 0) {
                cl::vector<cl::Event> decompressWaitList = {unmapEvents[i]};
                cl842DeviceDecompressor->decompress(commandQueue,
                                                    buffer, superblock_offset, chunksSize, cl::Buffer(nullptr),
                                                    buffer, superblock_offset, chunksSize, cl::Buffer(nullptr),
                                                    cl::Buffer(nullptr),
                                                    &decompressWaitList, &decompressEvents[i]);
            } else {
                decompressEvents[i] = unmapEvents[i];
            }
        }


        *startEvent = mapEvents.front();
        *endEvent = decompressEvents.back();
    } else {
#endif
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
        cl::UserEvent receiveEvent(context);
        std::shared_ptr<dcl::CLEventCompletable> mapDataCompletable(new dcl::CLEventCompletable(*startEvent));
        read(transferId, size, ptr, false, mapDataCompletable)
            ->setCallback(std::bind(&cl::UserEvent::setStatus, receiveEvent, std::placeholders::_1));
        /* Enqueue unmap buffer (implicit upload) */
        cl::vector<cl::Event> unmapWaitList = {receiveEvent};
        commandQueue.enqueueUnmapMemObject(buffer, ptr, &unmapWaitList, endEvent);
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE) && defined(LIB842_HAVE_OPENCL)
    }
#endif

#ifdef PROFILE_SEND_RECEIVE_BUFFER
    auto profile_times = new profile_send_receive_buffer_times(); // TODOXXX: Release memory
    profile_times->id = profile_next_id++;
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
        auto profile_times = ((profile_send_receive_buffer_times *)user_data);
        profile_times->end_time = std::chrono::steady_clock::now();
        dcl::util::Logger << dcl::util::Debug
                          << "(PROFILE) Receive with id " << profile_times->id << " of size " << profile_times->transfer_size
                          << " uploaded (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
            profile_times->end_time - profile_times->start_time).count() << std::endl;
    }, profile_times);
#endif
}

std::shared_ptr<DataSending> DataStream::write(
        dcl::transfer_id transfer_id,
        size_t size, const void *ptr, bool skip_compress_step,
        const std::shared_ptr<dcl::Completable> &trigger_event) {
#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        // START UBER HACK
        if (size > SUPERBLOCK_MAX_SIZE) {
            size_t num_superblocks = (size + SUPERBLOCK_MAX_SIZE - 1) / SUPERBLOCK_MAX_SIZE;
            std::shared_ptr<DataSending> callback;

            dcl::transfer_id split_transfer_id = transfer_id;
            for (size_t i = 0; i < num_superblocks; i++, next_transfer_id_uberhax(split_transfer_id)) {
                size_t superblock_offset = i * SUPERBLOCK_MAX_SIZE;
                size_t superblock_size = std::min(size - superblock_offset, SUPERBLOCK_MAX_SIZE);
                callback = write(
                    split_transfer_id, superblock_size,
                    static_cast<const uint8_t *>(ptr) + superblock_offset,
                    skip_compress_step, trigger_event);
            }

            return callback;
        }
        // END UBER HACK
    }
#endif

    auto write(std::make_shared<DataSending>(transfer_id, size, ptr, skip_compress_step));
    if (trigger_event != nullptr) {
        // If a event to wait was given, enqueue the write after the event callback
        // TODOXXX: Handle the case where the event to wait returns an error
        trigger_event->setCallback([this, write](cl_int status) {
            assert(status == CL_SUCCESS); // TODOXXX Handle errors
            enqueue_write(write);
        });
    } else {
        // If a event to wait was given, enqueue the write immediately
        enqueue_write(write);
    }
    return write;
}

void DataStream::enqueue_write(const std::shared_ptr<DataSending> &write) {
    std::unique_lock<std::mutex> lock(_writeq_mtx);
    if (_sending) {
        _writeq.push(write);
    } else {
        // start write loop
        _sending = true;
        lock.unlock();

        notify_write_transfer_id(new writeq_type({ write }));
    }
}

void DataStream::notify_write_transfer_id(writeq_type *writeq) {
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
        *_socket, boost::asio::buffer(write->transferId().data, write->transferId().size()),
        [this, writeq, write](const boost::system::error_code& ec, size_t bytes_transferred) {
            start_write(writeq);
        });
}

void DataStream::start_write(writeq_type *writeq) {
    auto& write = writeq->front();
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
        << "Start write of size " << write->size()
        << std::endl;
#endif
    write->onStart();
    /* TODO *Move* writeq through (i.e., into and out of) lambda capture
     * In C++14 this should be possible by generalized lambda captures as follows:
    boost_asio_async_write_with_sentinels(
            *_socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq{std::move(writeq)}](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(std::move(writeq), ec, bytes_transferred); });
     */

#ifdef IO_LINK_COMPRESSION
    if (is_io_link_compression_enabled()) {
        _write_io_channel_busy = false;
        _write_io_total_bytes_transferred = 0;
        _write_io_num_blocks_remaining = write->size() / NETWORK_BLOCK_SIZE;

        if (write->size() >= NETWORK_BLOCK_SIZE) {
            _compress_current_writeq = writeq;
            _compress_current_write = write;
            _compress_current_offset = 0;

            std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
            _compress_trigger = true;
            _compress_trigger_changed.notify_all();
        }

        try_write_next_compressed_block(writeq, write);
        return;
    }
#endif

    boost_asio_async_write_with_sentinels(
            *_socket, boost::asio::buffer(write->ptr(), write->size()),
            [this, writeq](const boost::system::error_code& ec, size_t bytes_transferred){
                    handle_write(writeq, ec, bytes_transferred); });
}

#ifdef IO_LINK_COMPRESSION
void DataStream::try_write_next_compressed_block(writeq_type *writeq, const std::shared_ptr<DataSending> &write) {
    std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
    if (_write_io_channel_busy) {
        // We're already inside a boost::asio::async_write call, so we can't initiate another one until it finishes
        return;
    }
    if (_write_io_num_blocks_remaining == SIZE_MAX) {
        // Last block was already written (calls to this function by compression threads can be spurious)
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
            send_buffers[2 + i] = boost::asio::buffer(block.datas[i].get(), block.sizes[i]);
        boost_asio_async_write_with_sentinels(*_socket, send_buffers,
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             assert(!ec);

             std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
             _write_io_channel_busy = false;
             _write_io_queue.pop();
             _write_io_total_bytes_transferred += bytes_transferred;
             _write_io_num_blocks_remaining--;
             lock.unlock();

             try_write_next_compressed_block(writeq, write);
         });
    } else {
        // Always write the last incomplete block of the input uncompressed
        _write_io_channel_busy = true;
        lock.unlock();


        auto last_block_source_ptr =  static_cast<const uint8_t *>(write->ptr()) + (write->size() & ~(NETWORK_BLOCK_SIZE - 1));
        auto last_block_size = write->size() & (NETWORK_BLOCK_SIZE - 1);

        boost_asio_async_write_with_sentinels(*_socket,
                boost::asio::buffer(last_block_source_ptr, last_block_size),
         [this, writeq, write](const boost::system::error_code &ec, size_t bytes_transferred) {
             assert(!ec);

             std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
             _write_io_channel_busy = false;
             _write_io_total_bytes_transferred += bytes_transferred;
             _write_io_num_blocks_remaining = SIZE_MAX;
             lock.unlock();

             // The data transfer thread also joins the final barrier for the compression
             // threads before finishing the write, to ensure resources are not released
             // while a compression thread still hasn't realized all work is finished
             if (write->size() >= NETWORK_BLOCK_SIZE) {
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
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
        << "Start compression thread with id " << thread_id
        << std::endl;
    size_t stat_handled_blocks = 0;
#endif
    while (true) {
        {
            std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
            _compress_trigger_changed.wait(lock, [this] { return _compress_trigger; });
            if (_compress_quit)
                break;
        }

        _compress_start_barrier.wait();
        _compress_trigger = false;

        auto writeq = _compress_current_writeq;
        auto write = _compress_current_write;
        auto last_valid_offset = write->size() & ~(NETWORK_BLOCK_SIZE-1);

        while (true) {
            size_t offset = _compress_current_offset.fetch_add(NETWORK_BLOCK_SIZE);
            if (offset >= last_valid_offset) {
                break;
            }

#ifdef INDEPTH_TRACE
            stat_handled_blocks++;
#endif

            write_block block;
            block.source_offset = offset;
            for (size_t i = 0; i < NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                auto source = static_cast<const uint8_t *>(write->ptr()) + offset + i * CHUNK_SIZE;

                if (write->skip_compress_step()) {
                    auto is_compressed = std::equal(source,source + sizeof(CL842_COMPRESSED_CHUNK_MAGIC), CL842_COMPRESSED_CHUNK_MAGIC);

                    auto chunk_buffer_size = is_compressed
                                             ? *reinterpret_cast<const uint64_t *>((source + sizeof(CL842_COMPRESSED_CHUNK_MAGIC)))
                                             : CHUNK_SIZE;
                    auto chunk_buffer = is_compressed
                            ? source + CHUNK_SIZE - chunk_buffer_size
                            : source;

                    block.datas[i] = std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter>(
                                chunk_buffer, ConditionalOwnerDeleter(false));
                    block.sizes[i] = chunk_buffer_size;
                } else {
                    // Compress chunk
                    // TODOXXX Should probably avoid doing separate allocations for chunks in a network block
                    std::unique_ptr<uint8_t[]> compress_buffer(new uint8_t[2 * CHUNK_SIZE]);
                    size_t compressed_size = 2 * CHUNK_SIZE;

                    int ret = lib842_compress(source, CHUNK_SIZE, compress_buffer.get(),
                                              &compressed_size);
                    assert(ret == 0);

                    // Push into the chunk queue
                    auto compressible = compressed_size <= COMPRESSIBLE_THRESHOLD;

                    // If the chunk is compressible, transfer ownership of the compressed buffer,
                    // otherwise use the uncompressed buffer and destroy the compressed buffer
                    block.datas[i] = std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter>(
                                    compressible ? compress_buffer.release() : source,
                                    ConditionalOwnerDeleter(compressible));
                    block.sizes[i] = compressible ? compressed_size : CHUNK_SIZE;
                }

            }

            {
                std::unique_lock<std::mutex> lock(_write_io_queue_mutex);
                _write_io_queue.push(std::move(block));
            }

            try_write_next_compressed_block(writeq, write);
        }

        _compress_finish_barrier.wait();
    }

#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
        << "End compression thread with id " << thread_id << " (stat_handled_blocks=" << stat_handled_blocks << ")"
        << std::endl;
#endif
}
#endif

void DataStream::handle_write(
        writeq_type *writeq,
        const boost::system::error_code& ec,
        size_t bytes_transferred) {
    // current write is first element in writeq, so writeq must be non-empty
    assert(writeq /* ouch! */ && !writeq->empty());
    auto& write = writeq->front();
#ifdef INDEPTH_TRACE
    dcl::util::Logger << dcl::util::Debug
        << "(DataStream to " << _remote_endpoint << ") "
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

void DataStream::writeFromClBuffer(
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
    auto profile_times = new profile_send_receive_buffer_times(); // TODOXXX: Release memory
    profile_times->id = profile_next_id++;
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
        auto profile_times = ((profile_send_receive_buffer_times *)user_data);
        profile_times->end_time = std::chrono::steady_clock::now();
        dcl::util::Logger << dcl::util::Debug
            << "(PROFILE) Send with id " << profile_times->id << " of size " << profile_times->transfer_size
            << " uploaded (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    profile_times->end_time - profile_times->start_time).count() << std::endl;
    }, profile_times);
#endif
}

} // namespace comm

} // namespace dclasio
