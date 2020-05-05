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
 * \file DataCompressionWorkPool.cpp
 *
 * \date 2020-05-05
 * \author Joan Bruguera Micó
 */

#ifdef IO_LINK_COMPRESSION

#include <dcl/util/Logger.h>

#include <sw842.h>
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
// TODOXXX: Should add the code to spread the threads among NUMA zones? (From lib842 sample)
#include <hw842.h>
#endif

#include "DataCompressionWorkPool.h"

// If INDEPTH_TRACE is defined, more traces and statistics are generated
//#define INDEPTH_TRACE

static int lib842_compress(const uint8_t *in, size_t ilen,
                           uint8_t *out, size_t *olen) {
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
    if (is_hw_io_link_compression_enabled())
        return hw842_compress(in ,ilen, out, olen);
#endif

    return optsw842_compress(in, ilen, out, olen);
}

namespace dclasio {

namespace comm {

DataCompressionWorkPool::DataCompressionWorkPool() :
    _compress_threads(determine_io_link_compression_num_threads("DCL_IO_LINK_NUM_COMPRESS_THREADS")),
    _compress_trigger(false), _compress_quit(false),
    _compress_ptr(nullptr), _compress_size(0), _compress_skip_compress_step(false),
    _compress_current_offset(0),
    _compress_start_barrier(_compress_threads.size()),
    _compress_finish_barrier(_compress_threads.size()+1) {
    for (size_t i = 0; i < _compress_threads.size(); i++) {
        _compress_threads[i] = std::thread{&DataCompressionWorkPool::loop_compress_thread, this, i};
    }
}

DataCompressionWorkPool::~DataCompressionWorkPool() {
    {
        std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
        _compress_trigger = true;
        _compress_quit = true;
        _compress_trigger_changed.notify_all();
    }
    for (auto &t : _compress_threads)
        t.join();
}

void DataCompressionWorkPool::start(
    const void *ptr, size_t size, bool skip_compress_step,
    std::function<void(write_block &&)> block_available_callback) {
    _compress_block_available_callback = std::move(block_available_callback);
    _compress_ptr = ptr;
    _compress_size = size;
    _compress_skip_compress_step = skip_compress_step;
    _compress_current_offset = 0;

    std::unique_lock<std::mutex> lock(_compress_trigger_mutex);
    _compress_trigger = true;
    _compress_trigger_changed.notify_all();
}

void DataCompressionWorkPool::finish() {
    _compress_finish_barrier.wait();
}

void DataCompressionWorkPool::loop_compress_thread(size_t thread_id) {
    static constexpr size_t CHUNK_SIZE = dcl::DataTransfer::COMPR842_CHUNK_SIZE;

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

        auto last_valid_offset = _compress_size & ~(dcl::DataTransfer::NETWORK_BLOCK_SIZE-1);

        while (true) {
            size_t offset = _compress_current_offset.fetch_add(dcl::DataTransfer::NETWORK_BLOCK_SIZE);
            if (offset >= last_valid_offset) {
                break;
            }

#ifdef INDEPTH_TRACE
            stat_handled_blocks++;
#endif

            write_block block;
            block.source_offset = offset;
            for (size_t i = 0; i < dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK; i++) {
                auto source = static_cast<const uint8_t *>(_compress_ptr) + offset + i * CHUNK_SIZE;

                if (_compress_skip_compress_step) {
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
                    auto compressible = compressed_size <= dcl::DataTransfer::COMPRESSIBLE_THRESHOLD;

                    // If the chunk is compressible, transfer ownership of the compressed buffer,
                    // otherwise use the uncompressed buffer and destroy the compressed buffer
                    block.datas[i] = std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter>(
                                    compressible ? compress_buffer.release() : source,
                                    ConditionalOwnerDeleter(compressible));
                    block.sizes[i] = compressible ? compressed_size : CHUNK_SIZE;
                }
            }

            _compress_block_available_callback(std::move(block));
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

} // namespace comm

} // namespace dclasio

#endif // IO_LINK_COMPRESSION
