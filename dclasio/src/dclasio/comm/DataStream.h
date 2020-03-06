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

#include <dcl/Completable.h>
#include <dcl/DCLTypes.h>

#include <boost/asio/ip/tcp.hpp>
#include <boost/variant.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <type_traits>

namespace dclasio {

namespace comm {

// TODO Split DataStream into InputDataStream and OutputDataStream to model simplex connections and reduce code redundancy
/*!
 * \brief A data stream maintains a set of incoming and outgoing data transfers from/to a single remote process
 */
class DataStream {
public:
    typedef std::queue<std::shared_ptr<DataReceipt>, std::list<std::shared_ptr<DataReceipt>>> readq_type;
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
            size_t size,
            void *ptr,
            bool skip_compress_step,
            const std::shared_ptr<dcl::Completable> &trigger_event);

    /*!
     * \brief Submits a data sending for this data stream
     *
     * \param[in]  size  number of bytes to send
     * \param[in]  ptr   source buffer for sent bytes
     * \return a handle for the data sending
     */
    std::shared_ptr<DataSending> write(
            size_t size,
            const void *ptr,
            bool skip_compress_step,
            const std::shared_ptr<dcl::Completable> &trigger_event);

private:
    /*!
     * \brief Processes the next data transfer from the read queue.
     *
     * \param[in]  readq    list of incoming data transfers to process
     *             If \c readq is empty, it is filled with data transfers from the stream's internal read queue
     */
    void schedule_read(readq_type *readq);
    void start_read(readq_type *readq);

#ifdef IO_LINK_COMPRESSION
    void read_next_compressed_chunk(readq_type *readq, std::shared_ptr<DataReceipt> read);
    void loop_decompress_thread();
#endif

    void handle_read(
            readq_type *readq,
            const boost::system::error_code& ec,
            size_t bytes_transferred);

    /*!
     * \brief Processes the next data transfer from the write queue.
     *
     * \param[in]  writeq   list of incoming data transfers to process
     *             If \c writeq is empty, it is filled with data transfers from the stream's internal write queue
     */
    void schedule_write(writeq_type *writeq);
    void start_write(writeq_type *writeq);

#ifdef IO_LINK_COMPRESSION
    void try_write_next_compressed_chunk(writeq_type *writeq, std::shared_ptr<DataSending> write);
    void loop_compress_thread();
#endif

    void handle_write(
            writeq_type *writeq,
            const boost::system::error_code& ec,
            size_t bytes_transferred);

    // TODO Store socket instance rather than smart pointer
    std::shared_ptr<boost::asio::ip::tcp::socket> _socket; //!< I/O object for remote process
    boost::asio::ip::tcp::endpoint _remote_endpoint; //!< remote endpoint of data stream

    bool _receiving; //!< \c true, if currently receiving data, otherwise \c false
    bool _sending; //!< \c true, if currently sending data, otherwise \c false
    readq_type _readq; //!< pending data receipts
    std::mutex _readq_mtx; //!< protects read queue and flag
    writeq_type _writeq; //!< pending data sendings
    std::mutex _writeq_mtx; //!< protects write queue and flag

#ifdef IO_LINK_COMPRESSION
    struct decompress_message_decompress {
        std::vector<uint8_t> compressed_data;
        void *destination;

        // Disable default copy constructor/assignment to prevent accidental performance hit
        decompress_message_decompress() = default;
        decompress_message_decompress(const decompress_message_decompress &) = delete;
        decompress_message_decompress& operator=(const decompress_message_decompress &) = delete;
        decompress_message_decompress(decompress_message_decompress &&) = default;
        decompress_message_decompress& operator=(decompress_message_decompress &&) = default;
    };
    struct decompress_message_finalize {
        readq_type *readq;
    };
    struct decompress_message_quit {};
    using decompress_message_t = boost::variant<decompress_message_decompress,
                                                decompress_message_finalize,
                                                decompress_message_quit>;

    // ** Variables related to the decompression thread (associated to reads) **
    // Instance of the decompression thread
    std::thread _decompress_thread;
    // Mutex for protecting concurrent accesses to
    // (_decompress_queue, _decompress_working_thread_count)
    std::mutex _decompress_queue_mutex;
    // Stores pending decompression operations after reads,
    // and can also receive other kinds of messages for lifetime management
    std::queue<decompress_message_t> _decompress_queue;
    // Wakes up the decompression threads when new operations have been added to the queue
    std::condition_variable _decompress_queue_available;
    // Number of threads currently running decompression operations
    unsigned _decompress_working_thread_count;

    // ** Variables related to the current asynchronous I/O read operation **
    // Total bytes transferred through the network by current read (for statistical purposes)
    size_t _read_io_total_bytes_transferred;
    // Number of full chunks remaining to transfer
    size_t _read_io_num_chunks_remaining;
    // Offset into the destination buffer where the data associated
    // with the current read operation will go (after decompression)
    size_t _read_io_destination_offset;
    // Size of the current read operation
    size_t _read_io_buffer_size;
    // Target buffer of the current read operation
    std::vector<uint8_t> _read_io_buffer;

    // A custom deleter for std::unique_ptr<const uint8_t[]> that conditionally deletes a value
    // If is_owner = true, the pointer owns the value (works like a regular std::unique_ptr)
    // If is_owner = false, it doesn't own the value (works like a raw pointer)
    // (in this case, the pointer must be kept alive by other means during the lifetime of the class)
    class ConditionalOwnerDeleter {
        bool is_owner;
    public:
        ConditionalOwnerDeleter(bool is_owner) : is_owner(is_owner) {}
        void operator()(const uint8_t *ptr)
        {
            if (is_owner) {
                delete[] ptr;
            }
        }
    };

    struct write_chunk {
        std::unique_ptr<const uint8_t[], ConditionalOwnerDeleter> data;
        size_t size;
    };

    // ** Variables related to the compression thread (associated to writes) **
    // Instance of the compression thread
    std::thread _compress_thread;
    // Mutex for protecting concurrent accesses to
    // (_compress_trigger, _compress_quit)
    std::mutex _compress_trigger_mutex;
    // true if a new operation must be started in the compression thread
    bool _compress_trigger = false;
    // Wakes up the compression thread when a new operation must be started
    std::condition_variable _compress_trigger_changed;
    // If set to true, causes the compression to quit (for cleanup)
    bool _compress_quit = false;
    // Necessary data for triggering an asynchronous I/O write operation from the compression thread
    writeq_type *_compress_current_writeq;
    std::shared_ptr<DataSending> _compress_current_write;

    // ** Variables related to the current asynchronous I/O write operation **
    // Total bytes transferred through the network by current write (for statistical purposes)
    size_t _write_io_total_bytes_transferred;
    // Offset into the source buffer where the data associated
    // with the current write operation comes from (before compression)
    size_t _write_io_source_offset;
    // Mutex for protecting concurrent accesses to
    // (_write_io_queue, _write_io_channel_busy)
    std::mutex _write_io_queue_mutex;
    // Stores pending write operations after compression
    std::queue<write_chunk> _write_io_queue;
    // Set when a write operation is in progress, so a new write operation knows it has to wait
    bool _write_io_channel_busy;
#endif
};

} // namespace comm

} // namespace dclasio

#endif /* DATASTREAM_H_ */
