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
 * \file DataCompressionWorkPool.h
 *
 * \date 2020-05-05
 * \author Joan Bruguera Micó
 */

#ifndef DATACOMPRESSIONWORKPOOL_H_
#define DATACOMPRESSIONWORKPOOL_H_

#ifdef IO_LINK_COMPRESSION

#include <dcl/DataTransfer.h>

#include <boost/thread/barrier.hpp>

#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include <array>
#include <vector>

#include <cstdint>
#include <memory>
#include <functional>

namespace dclasio {

namespace comm {

class DataCompressionWorkPool {
public:
    // A custom deleter for std::unique_ptr<const uint8_t[]> that conditionally deletes a value
    // If is_owner = true, the pointer owns the value (works like a regular std::unique_ptr)
    // If is_owner = false, it doesn't own the value (works like a raw pointer)
    // (in this case, the pointer must be kept alive by other means during the lifetime of the class)
    class ConditionalOwnerDeleter {
        bool is_owner;
    public:
        ConditionalOwnerDeleter() : is_owner(true) {}
        explicit ConditionalOwnerDeleter(bool is_owner) : is_owner(is_owner) {}
        void operator()(const uint8_t *ptr)
        {
            if (is_owner) {
                delete[] ptr;
            }
        }
    };

    struct write_block {
        // Offset into the source buffer where the data associated with the block comes from
        size_t source_offset;
        // Data for each (possibly compressed) chunk in the block
        std::array<std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter>,
                   dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK> datas;
        // Size for each (possibly compressed) chunk in the block
        std::array<size_t, dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK> sizes;
    };

    DataCompressionWorkPool();
    ~DataCompressionWorkPool();

    void start(const void *ptr, size_t size, bool skip_compress_step,
               std::function<void(write_block &&)> block_available_callback);
    void finish();

private:
    void loop_compress_thread(size_t thread_id);

    // Instance of the compression thread
    std::vector<std::thread> _compress_threads;
    // Mutex for protecting concurrent accesses to
    // (_compress_trigger, _compress_quit)
    std::mutex _compress_trigger_mutex;
    // true if a new operation must be started in the compression thread
    bool _compress_trigger;
    // Wakes up the compression thread when a new operation must be started
    std::condition_variable _compress_trigger_changed;
    // If set to true, causes the compression to quit (for cleanup)
    bool _compress_quit;
    // Necessary data for triggering an asynchronous I/O write operation from the compression thread
    std::function<void(write_block &&)> _compress_block_available_callback;
    // Data for the compression operation in course
    const void *_compress_ptr;
    size_t _compress_size;
    bool _compress_skip_compress_step;
    std::atomic<std::size_t> _compress_current_offset;
    // Barrier for starting compression, necessary for ensuring that all compression
    // threads have seen the trigger to start compressing before unsetting it
    boost::barrier _compress_start_barrier;
    // Barrier for finishing compression, necessary for ensuring that resources
    // are not released until all threads have finished
    boost::barrier _compress_finish_barrier;
};

} // namespace comm

} // namespace dclasio

#endif // IO_LINK_COMPRESSION

#endif // DATACOMPRESSIONWORKPOOL_H_
