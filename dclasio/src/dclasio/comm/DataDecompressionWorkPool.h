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
 * \file DataDecompressionWorkPool.h
 *
 * \date 2020-05-05
 * \author Joan Bruguera Micó
 */

#ifndef DATADECOMPRESSIONWORKPOOL_H_
#define DATADECOMPRESSIONWORKPOOL_H_

#ifdef IO_LINK_COMPRESSION

#include <dcl/DataTransfer.h>

#include <boost/thread/barrier.hpp>
#include <boost/variant.hpp>

#include <condition_variable>
#include <mutex>
#include <thread>

#include <functional>
#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <queue>

namespace dclasio {

namespace comm {

class DataDecompressionWorkPool {
public:
    struct decompress_chunk {
        std::vector<uint8_t> compressed_data;
        void *destination;

        // Disable default copy constructor/assignment to prevent accidental performance hit
        decompress_chunk() = default;
        decompress_chunk(const decompress_chunk &) = delete;
        decompress_chunk& operator=(const decompress_chunk &) = delete;
        decompress_chunk(decompress_chunk &&) = default;
        decompress_chunk& operator=(decompress_chunk &&) = default;
    };

    struct decompress_message_decompress_block {
        std::array<decompress_chunk, dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK> chunks;
    };

    DataDecompressionWorkPool();
    ~DataDecompressionWorkPool();
    void start();
    void push_block(DataDecompressionWorkPool::decompress_message_decompress_block &&dm);
    void finalize(std::function<void()> finalize_callback);

private:
    void loop_decompress_thread(size_t thread_id);

    struct decompress_message_finalize {
        std::function<void()> finalize_callback;
    };
    struct decompress_message_quit {};
    using decompress_message_t = boost::variant<decompress_message_decompress_block,
        decompress_message_finalize,
        decompress_message_quit>;

    // Instance of the decompression threads
    std::vector<std::thread> _decompress_threads;
    // Mutex for protecting concurrent accesses to
    // (_decompress_queue, _decompress_working_thread_count)
    std::mutex _decompress_queue_mutex;
    // Stores pending decompression operations after reads,
    // and can also receive other kinds of messages for lifetime management
    std::queue<decompress_message_t> _decompress_queue;
    // Wakes up the decompression threads when new operations have been added to the queue
    std::condition_variable _decompress_queue_available;
    // Number of threads currently running decompression operations
    unsigned int _decompress_working_thread_count;
    // Barrier for finishing decompression, necessary for ensuring that resources
    // are not released until all threads have finished
    boost::barrier _decompress_finish_barrier;
};

} // namespace comm

} // namespace dclasio

#endif // IO_LINK_COMPRESSION

#endif // DATADECOMPRESSIONWORKPOOL_H_
