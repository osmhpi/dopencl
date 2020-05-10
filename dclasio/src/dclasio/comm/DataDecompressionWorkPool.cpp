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
 * \file DataDecompressionWorkPool.cpp
 *
 * \date 2020-05-05
 * \author Joan Bruguera Micó
 */

#ifdef IO_LINK_COMPRESSION

#include <sw842.h>
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
// TODOXXX: Should add the code to spread the threads among NUMA zones? (From lib842 sample)
#include <hw842.h>
#endif

#include "DataDecompressionWorkPool.h"

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

static int lib842_decompress(const uint8_t *in, size_t ilen,
                             uint8_t *out, size_t *olen) {
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
    if (is_hw_io_link_compression_enabled())
        return hw842_decompress(in ,ilen, out, olen);
#endif

    return optsw842_decompress(in, ilen, out, olen);
}

namespace dclasio {

namespace comm {

DataDecompressionWorkPool::DataDecompressionWorkPool(
    std::function<std::ostream&(void)> error_logger,
    std::function<std::ostream&(void)> debug_logger) :
    _error_logger(std::move(error_logger)),
    _debug_logger(std::move(debug_logger)),
    _threads(determine_io_link_compression_num_threads("DCL_IO_LINK_NUM_DECOMPRESS_THREADS")),
    _state(decompress_state::processing),
    _working_thread_count(0),
    _finish_barrier(_threads.size()),
    _report_error(false) {

    for (size_t i = 0; i < _threads.size(); i++) {
        _threads[i] = std::thread{&DataDecompressionWorkPool::loop_decompress_thread, this, i};
    }
}

DataDecompressionWorkPool::~DataDecompressionWorkPool() {
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        _state = decompress_state::quitting;
        _queue_available.notify_all();
    }
    for (auto &t : _threads)
        t.join();
}

void DataDecompressionWorkPool::start() {
    _working_thread_count = 0;
}

bool DataDecompressionWorkPool::push_block(DataDecompressionWorkPool::decompress_block &&dm) {
    std::lock_guard<std::mutex> lock(_queue_mutex);
    if (_report_error) {
        _report_error = false;
        return false;
    }

    _queue.push(std::move(dm));
    _queue_available.notify_one();
    return true;
}

void DataDecompressionWorkPool::finalize(bool cancel, const std::function<void(bool)> &finalize_callback) {
    bool done, report_error;
    {
        std::lock_guard<std::mutex> lock(_queue_mutex);
        done = _state == decompress_state::processing &&
               _working_thread_count == 0 && _queue.empty();
        // If there are still decompression operations active, we need to wait
        // until they finish to finalize the entire operation
        // In this case, transfer the responsibility of finalizing to the decompression threads
        if (!done) {
            _state = cancel ? decompress_state::cancelling : decompress_state::finalizing;
            _finalize_callback = finalize_callback;
            _queue_available.notify_all();
        }

        report_error = _report_error;
        _report_error = false;
    }

    // Otherwise, if all decompression threads finished as well,
    // finalize the entire operation as soon as possible here
    if (done) {
        finalize_callback(!report_error);
    }
}

void DataDecompressionWorkPool::loop_decompress_thread(size_t thread_id) {
#ifdef INDEPTH_TRACE
    _debug_logger()
        << "(DataStream to " << _remote_endpoint << ") "
        << "Start decompression thread with id " << thread_id
        << std::endl;
    size_t stat_handled_blocks = 0;
#endif

    while (true) {
        // (Blocking) pop from the chunk queue
        std::unique_lock<std::mutex> lock(_queue_mutex);
        _queue_available.wait(lock, [this] {
            return !_queue.empty() || _state != decompress_state::processing;
        });
        if (_state == decompress_state::handling_error || _state == decompress_state::cancelling) {
            lock.unlock();

            // Wait until all threads have got the "error" message
            _finish_barrier.wait();

            // "Leader" thread clears the queue
            if (thread_id == 0) {
                lock.lock();
                _queue = {};
                _state = decompress_state::processing;
                _report_error = _state == decompress_state::handling_error;
                lock.unlock();
            }

            // Once write is finalized, wait again
            _finish_barrier.wait();
        } else if (_state == decompress_state::finalizing && _queue.empty()) {
            lock.unlock();

            // Wait until all threads have got the "finalize" message
            _finish_barrier.wait();

            // "Leader" thread finalizes the write
            if (thread_id == 0) {
                lock.lock();
                auto report_error = _report_error;
                _report_error = false;
                _state = decompress_state::processing;
                lock.unlock();
                _finalize_callback(!report_error);
            }

            // Once write is finalized, wait again
            _finish_barrier.wait();
        } else if (_state == decompress_state::quitting) {
            break;
        } else {
            auto block = std::move(_queue.front());
            _queue.pop();
            _working_thread_count++;

            lock.unlock();
#ifdef INDEPTH_TRACE
            stat_handled_blocks++;
#endif
            for (size_t i = 0; i < dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                const auto &chunk = block.chunks[i];
                if (chunk.compressed_data == nullptr && chunk.compressed_length == 0 &&
                    chunk.destination == nullptr) {
                    // Chunk was transferred uncompressed, nothing to do
                    continue;
                }


                auto destination = static_cast<uint8_t *>(chunk.destination);

                assert(chunk.compressed_length > 0 &&
                       chunk.compressed_length <= dcl::DataTransfer::COMPRESSIBLE_THRESHOLD);

                size_t uncompressed_size = dcl::DataTransfer::COMPR842_CHUNK_SIZE;
                int ret = lib842_decompress(chunk.compressed_data,
                                            chunk.compressed_length,
                                            destination, &uncompressed_size);
                if (ret != 0) {
                    _error_logger()
                            << "Data decompression failed, aborting operation"
                            << std::endl;

                    lock.lock();
                    _state = decompress_state::handling_error;
                    _queue_available.notify_all();
                    lock.unlock();
                    break;
                }

                assert(uncompressed_size == dcl::DataTransfer::COMPR842_CHUNK_SIZE);
            }

            lock.lock();
            _working_thread_count--;
            lock.unlock();
        }
    }

#ifdef INDEPTH_TRACE
    _debug_logger()
        << "(DataStream to " << _remote_endpoint << ") "
        << "End decompression thread with id " << thread_id << " (stat_handled_blocks=" << stat_handled_blocks << ")"
        << std::endl;
#endif
}

} // namespace comm

} // namespace dclasio

#endif // IO_LINK_COMPRESSION
