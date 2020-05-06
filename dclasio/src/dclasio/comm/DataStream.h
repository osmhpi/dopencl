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
 * \file DataStream.h
 *
 * \date 2014-03-07
 * \author Philipp Kegel
 */

#ifndef DATASTREAM_H_
#define DATASTREAM_H_

#include "DataTransferImpl.h"
#include "DataDecompressionWorkPool.h"
#include "DataCompressionWorkPool.h"

#include <dcl/Completable.h>
#include <dcl/DCLTypes.h>

#include <boost/asio/ip/tcp.hpp>

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE)
#include <cl842.h>
#endif

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>

namespace dclasio {

namespace comm {

// TODO Split DataStream into InputDataStream and OutputDataStream to model simplex connections and reduce code redundancy
/*!
 * \brief A data stream maintains a set of incoming and outgoing data transfers from/to a single remote process
 */
class DataStream {
public:
    typedef std::queue<std::shared_ptr<DataSending>, std::list<std::shared_ptr<DataSending>>> writeq_type;

    // TODO Accept rvalue reference rather than pointer to socket (requires Boost 1.47)
    /*!
     * \brief Creates a data stream from a connected socket
     * The data stream becomes owner of the socket.
     *
     * \param[in]  socket   the socket to use for the data stream
     */
    DataStream(
            const std::shared_ptr<boost::asio::ip::tcp::socket>& socket);
    /*!
     * \brief Creates a data stream to the specified remote endpoint
     *
     * \param socket            a socket associated with a local endpoint
     * \param remote_endpoint   the remote process
     */
    DataStream(
            const std::shared_ptr<boost::asio::ip::tcp::socket>& socket,
            boost::asio::ip::tcp::endpoint remote_endpoint);
    virtual ~DataStream();

    /* Data streams must be non-copyable */
    DataStream(const DataStream&) = delete;
    DataStream& operator=(const DataStream&) = delete;

    /*!
     * \brief Connects this data stream to its remote process
     * The ID of the local process associated with this data stream is send to
     * the remote process.
     *
     * \param[in]  pid  ID of the local process
     * \return the ID of the remote process, or 0 if the connection has been rejected
     */
    dcl::process_id connect(
            dcl::process_id pid);

    void disconnect();

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
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE)
            const CL842DeviceDecompressor *cl842DeviceDecompressor,
#endif
            const cl::CommandQueue &commandQueue,
            const cl::Buffer &buffer,
            size_t offset,
            const cl::vector<cl::Event> *eventWaitList,
            cl::Event *startEvent,
            cl::Event *endEvent);

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
            const cl::Context &context,
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
    std::shared_ptr<boost::asio::ip::tcp::socket> _socket; //!< I/O object for remote process
    boost::asio::ip::tcp::endpoint _remote_endpoint; //!< remote endpoint of data stream

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

    bool _sending; //!< \c true, if currently sending data, otherwise \c false
    writeq_type _writeq; //!< pending data sendings
    std::mutex _writeq_mtx; //!< protects write queue and flag

#ifdef IO_LINK_COMPRESSION
    static constexpr size_t NUM_CHUNKS_PER_NETWORK_BLOCK = dcl::DataTransfer::NUM_CHUNKS_PER_NETWORK_BLOCK;
    static constexpr size_t CHUNK_SIZE = dcl::DataTransfer::COMPR842_CHUNK_SIZE;
    static constexpr size_t COMPRESSIBLE_THRESHOLD = dcl::DataTransfer::COMPRESSIBLE_THRESHOLD;
    static constexpr size_t NETWORK_BLOCK_SIZE = dcl::DataTransfer::NETWORK_BLOCK_SIZE;
    static constexpr size_t SUPERBLOCK_MAX_SIZE = static_cast<size_t>(1) << 29; // 512 MiB
    // ---

    // ** Variables related to the decompression thread (associated to reads) **
    std::unique_ptr<DataDecompressionWorkPool> _decompress_thread_pool;

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
    std::array<std::vector<uint8_t>, NUM_CHUNKS_PER_NETWORK_BLOCK> _read_io_buffers;

    // ** Variables related to the compression thread (associated to writes) **
    std::unique_ptr<DataCompressionWorkPool> _compress_thread_pool;

    // ** Variables related to the current asynchronous I/O write operation **
    // Total bytes transferred through the network by current write (for statistical purposes)
    size_t _write_io_total_bytes_transferred;
    // Number of network blocks remaining to transfer
    size_t _write_io_num_blocks_remaining;
    // Mutex for protecting concurrent accesses to
    // (_write_io_queue, _write_io_channel_busy)
    std::mutex _write_io_queue_mutex;
    // Stores pending write operations after compression
    std::queue<DataCompressionWorkPool::write_block> _write_io_queue;
    bool _write_io_compression_error;
    // Set when a write operation is in progress, so a new write operation knows it has to wait
    bool _write_io_channel_busy;
#endif
};

} // namespace comm

} // namespace dclasio

#endif /* DATASTREAM_H_ */
