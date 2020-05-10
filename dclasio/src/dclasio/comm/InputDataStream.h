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
 * \file InputDataStream.h
 *
 * \date 2014-03-07
 * \author Philipp Kegel
 */

#ifndef INPUTDATASTREAM_H_
#define INPUTDATASTREAM_H_

#include "DataTransferImpl.h"

#include <dcl/Completable.h>
#include <dcl/DCLTypes.h>

#include <lib842/stream/decomp.h>

#include <boost/asio/ip/tcp.hpp>

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION)
#include <lib842/cl.h>
#endif

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <cstddef>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace dclasio {

namespace comm {

class InputDataStream {
public:
    InputDataStream(boost::asio::ip::tcp::socket& socket);
    virtual ~InputDataStream() = default;

    /* Data streams must be non-copyable */
    InputDataStream(const InputDataStream&) = delete;
    InputDataStream& operator=(const InputDataStream&) = delete;

    /*!
     * \brief Submits a data receipt for this data stream
     *
     * \param[in]  size  number of bytes to receipt
     * \param[in]  ptr   destination buffer for received bytes
     * \return a handle for the data receipt
     */
    std::shared_ptr<DataReceipt> read(
            dcl::transfer_id transfer_id,
            size_t size,
            void *ptr,
            bool skip_compress_step,
            const std::shared_ptr<dcl::Completable> &trigger_event);

    void readToClBuffer(
            dcl::transfer_id transferId,
            size_t size,
            const cl::Context &context,
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
            const lib842::CLDeviceDecompressor *cl842DeviceDecompressor,
#endif
            const cl::CommandQueue &commandQueue,
            const cl::Buffer &buffer,
            size_t offset,
            const cl::vector<cl::Event> *eventWaitList,
            cl::Event *startEvent,
            cl::Event *endEvent);

private:
    void enqueue_read(const std::shared_ptr<DataReceipt> &read);
    void receive_matching_transfer_id();
    /*!
     * \brief Processes the next data transfer from the read queue.
     */
    void start_read();
#ifdef IO_LINK_COMPRESSION
    void read_next_compressed_block();
#endif
    void handle_read(
            const boost::system::error_code& ec,
            size_t bytes_transferred);

    boost::asio::ip::tcp::socket &_socket; //!< I/O object for remote process

    enum class receiving_state {
        // No read operation is in course and the read queue is empty
        idle,
        // We are receiving a transfer ID from the other end of the socket,
        // in order to attempt to match a send with out of the enqueued receives
        receiving_matching_transfer_id,
        // The other end wants to do a send for which we don't yet have a receive
        // operation associated, so we are waiting for it to arrive to the queue
        waiting_for_read_matching_transfer_id,
        // We have successfully matched a send and a receive and are transferring data
        receiving_data
    };

    receiving_state _read_state; //!< \c state of the data receipts
    dcl::transfer_id _read_transfer_id; //!< identifier of the transfer currently being received
                                        // valid for receiving_state::waiting_for_read_matching_transfer_id
    std::shared_ptr<DataReceipt> _read_op; //!< transfer currently being received
                                           // valid for receiving_state::receiving_data
    std::unordered_map<dcl::transfer_id, std::shared_ptr<DataReceipt>> _readq; //!< pending data receipts
    std::mutex _readq_mtx; //!< protects read queue and related variables

#ifdef IO_LINK_COMPRESSION
    static constexpr size_t NUM_CHUNKS_PER_NETWORK_BLOCK = lib842::stream::NUM_CHUNKS_PER_NETWORK_BLOCK;
    static constexpr size_t CHUNK_SIZE = lib842::stream::COMPR842_CHUNK_SIZE;
    static constexpr size_t COMPRESSIBLE_THRESHOLD = lib842::stream::COMPRESSIBLE_THRESHOLD;
    static constexpr size_t NETWORK_BLOCK_SIZE = lib842::stream::NETWORK_BLOCK_SIZE;
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    static constexpr size_t CL_UPLOAD_BLOCK_SIZE = dcl::DataTransfer::CL_UPLOAD_BLOCK_SIZE;
#endif
    // ---

    // ** Variables related to the decompression thread (associated to reads) **
    std::unique_ptr<lib842::stream::DataDecompressionStream> _decompress_thread_pool;

    // ** Variables related to the current asynchronous I/O read operation **
    // Total bytes transferred through the network by current read (for statistical purposes)
    size_t _read_io_total_bytes_transferred;
    // Number of network blocks remaining to transfer
    size_t _read_io_num_blocks_remaining;
    // Offset into the destination buffer where the data associated
    // with the current read operation will go (after decompression)
    size_t _read_io_destination_offset;
    // Size of the current read operation
    std::array<size_t, NUM_CHUNKS_PER_NETWORK_BLOCK> _read_io_buffer_sizes;
    // Target buffer of the current read operation
    std::unique_ptr<uint8_t> _read_io_compressed_buffer;
#endif
};

} // namespace comm

} // namespace dclasio

#endif /* INPUTDATASTREAM_H_ */
