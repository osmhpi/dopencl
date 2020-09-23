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

#include "InputDataStream.h"
#include "OutputDataStream.h"

#include <dcl/DCLTypes.h>

#include <boost/asio/ip/tcp.hpp>

#include <memory>

namespace dclasio {

namespace comm {

/*!
 * \brief A data stream maintains a set of incoming and outgoing data transfers from/to a single remote process
 */
class DataStream : public InputDataStream, public OutputDataStream {
public:
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
    ~DataStream() override = default;

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

private:
    // TODO Store socket instance rather than smart pointer
    std::shared_ptr<boost::asio::ip::tcp::socket> _socket; //!< I/O object for remote process
    boost::asio::ip::tcp::endpoint _remote_endpoint; //!< remote endpoint of data stream
};

} // namespace comm

} // namespace dclasio

#endif /* DATASTREAM_H_ */
