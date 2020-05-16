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
 * \file OutputDataStream.h
 *
 * \date 2014-03-07
 * \author Philipp Kegel
 */

#ifndef OUTPUTDATASTREAM_H_
#define OUTPUTDATASTREAM_H_

#include "DataTransferImpl.h"

#include <dcl/Completable.h>
#include <dcl/DCLTypes.h>

#ifdef IO_LINK_COMPRESSION
#include <lib842/stream/comp.h>
#endif

#include <boost/asio/ip/tcp.hpp>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <queue>

namespace dclasio {

namespace comm {

class OutputDataStream {
public:
    // TODO Accept rvalue reference rather than pointer to socket (requires Boost 1.47)
    OutputDataStream(boost::asio::ip::tcp::socket& socket);
    virtual ~OutputDataStream() = default;

    /* Data streams must be non-copyable */
    OutputDataStream(const OutputDataStream&) = delete;
    OutputDataStream& operator=(const OutputDataStream&) = delete;

    /*!
     * \brief Submits a data sending for this data stream
     *
     * \param[in]  size  number of bytes to send
     * \param[in]  ptr   source buffer for sent bytes
     * \return a handle for the data sending
     */
    std::shared_ptr<DataSending> write(
            dcl::transfer_id transfer_id,
            size_t size,
            const void *ptr,
            bool skip_compress_step,
            const std::shared_ptr<dcl::Completable> &trigger_event);

    void writeFromClBuffer(
            dcl::transfer_id transferId,
            size_t size,
            const dcl::CLOutDataTransferContext &clDataTransferContext,
            const cl::Buffer &buffer,
            size_t offset,
            const cl::vector<cl::Event> *eventWaitList,
            cl::Event *startEvent,
            cl::Event *endEvent);

private:
    typedef std::queue<std::shared_ptr<DataSending>, std::list<std::shared_ptr<DataSending>>> writeq_type;

    void enqueue_write(const std::shared_ptr<DataSending> &write);
    void notify_write_transfer_id(writeq_type *writeq);
    /*!
     * \brief Processes the next data transfer from the write queue.
     *
     * \param[in]  writeq   list of incoming data transfers to process
     *             If \c writeq is empty, it is filled with data transfers from the stream's internal write queue
     */
    void start_write(writeq_type *writeq);
#ifdef IO_LINK_COMPRESSION
    void try_write_next_compressed_block(writeq_type *writeq, const std::shared_ptr<DataSending> &write);
#endif
    void handle_write(
            writeq_type *writeq,
            const boost::system::error_code& ec,
            size_t bytes_transferred);

    // TODO Store socket instance rather than smart pointer
    boost::asio::ip::tcp::socket &_socket; //!< I/O object for remote process

    bool _sending; //!< \c true, if currently sending data, otherwise \c false
    writeq_type _writeq; //!< pending data sendings
    std::mutex _writeq_mtx; //!< protects write queue and flag

#ifdef IO_LINK_COMPRESSION
    // ** Variables related to the compression thread (associated to writes) **
    std::unique_ptr<lib842::stream::DataCompressionStream> _compress_thread_pool;

    // ** Variables related to the current asynchronous I/O write operation **
    // Total bytes transferred through the network by current write (for statistical purposes)
    size_t _write_io_total_bytes_transferred;
    // Number of network blocks remaining to transfer
    size_t _write_io_num_blocks_remaining;
    // Mutex for protecting concurrent accesses to
    // (_write_io_queue, _write_io_channel_busy)
    std::mutex _write_io_queue_mutex;
    // Stores pending write operations after compression
    std::queue<lib842::stream::DataCompressionStream::compress_block> _write_io_queue;
    bool _write_io_compression_error;
    // Set when a write operation is in progress, so a new write operation knows it has to wait
    bool _write_io_channel_busy;
#endif
};

} // namespace comm

} // namespace dclasio

#endif /* OUTPUTDATASTREAM_H_ */
