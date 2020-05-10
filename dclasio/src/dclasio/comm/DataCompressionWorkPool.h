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
#include <ostream>

namespace dclasio {

namespace comm {

class DataCompressionWorkPool {
public:
    struct compress_block {
        // Offset into the source buffer where the data associated with the block comes from
        size_t source_offset;
        // Data for each (possibly compressed) chunk in the block
        std::array<const uint8_t *, dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK> datas;
        // Size for each (possibly compressed) chunk in the block
        std::array<size_t, dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK> sizes;

        // Buffer that owns the pointers used in 'datas'. Used internally.
        std::unique_ptr<uint8_t[]> compress_buffer;
    };

    DataCompressionWorkPool(std::function<std::ostream&(void)> error_logger,
                            std::function<std::ostream&(void)> debug_logger);
    ~DataCompressionWorkPool();

    void start(const void *ptr, size_t size, bool skip_compress_step,
               std::function<void(compress_block &&)> block_available_callback);
    void finish(bool cancel);

private:
    void loop_compress_thread(size_t thread_id);

    std::function<std::ostream&(void)> _error_logger;
    std::function<std::ostream&(void)> _debug_logger;

    // Instance of the compression thread
    std::vector<std::thread> _threads;
    // Mutex for protecting concurrent accesses to
    // (_trigger, _quit)
    std::mutex _trigger_mutex;
    // true if a new operation must be started in the compression thread
    bool _trigger;
    // Wakes up the compression thread when a new operation must be started
    std::condition_variable _trigger_changed;
    // If set to true, causes the compression to quit (for cleanup)
    bool _quit;
    // Necessary data for triggering an asynchronous I/O write operation from the compression thread
    std::function<void(compress_block &&)> _block_available_callback;
    // Parameters for the compression operation in course
    const void *_ptr;
    size_t _size;
    bool _skip_compress_step;
    // Stores the offset of the next block to be compressed
    std::atomic<std::size_t> _current_offset;
    // true if an error has happened and the compression operation should be cancelled, false otherwise
    std::atomic<bool> _error;
    // Barrier for starting compression, necessary for ensuring that all compression
    // threads have seen the trigger to start compressing before unsetting it
    boost::barrier _start_barrier;
    // Barrier for finishing compression, necessary for ensuring that resources
    // are not released until all threads have finished
    boost::barrier _finish_barrier;
};

} // namespace comm

} // namespace dclasio

#endif // IO_LINK_COMPRESSION

#endif // DATACOMPRESSIONWORKPOOL_H_
