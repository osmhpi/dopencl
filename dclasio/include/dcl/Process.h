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
 * \file Process.h
 *
 * \date 2011-08-07
 * \author Philipp Kegel
 *
 * C++ API declarations for dOpenCL communication layer
 */

#ifndef DCL_PROCESS_H_
#define DCL_PROCESS_H_

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

/* TODO Remove message headers from process interface */
#include <dclasio/message/Message.h>

#include <dcl/Completable.h>
#include <dcl/DCLTypes.h>

#include <cstddef>
#include <memory>
#include <string>

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION_INPLACE)
#include <cl842.h>
#endif

namespace dcl {

class DataTransfer;

/* ****************************************************************************/

/*!
 * \brief A generalization of a node in a dOpenCL network.
 *
 * A process can represent a host or a compute node.
 */
class Process {
public:
    virtual ~Process() { }

    virtual const std::string& url() const = 0;

    /*!
     * \brief Sends a message to this process
     * \param[in]  message  the message to send
     *
     * \deprecated Messages are not part of the dOpenCL API and must not be send by the application layer directly
     */
    virtual void sendMessage(
            const dclasio::message::Message& message) const = 0;

	/*!
	 * \brief Send data to process.
	 * This is a non-blocking operation.
	 *
	 * \param[in]  size      size of data buffer
	 * \param[in]  ptr       data buffer
	 * \return
	 */
	virtual std::shared_ptr<DataTransfer> sendData(
			dcl::transfer_id transfer_id,
			size_t      size,
			const void *ptr,
			bool skip_compress_step = false,
			const std::shared_ptr<Completable> &trigger_event = nullptr) = 0;

	virtual void sendDataFromClBuffer(
			dcl::transfer_id transferId,
			size_t size,
			const cl::Context &context,
			const cl::CommandQueue &commandQueue,
			const cl::Buffer &buffer,
			size_t offset,
			const cl::vector<cl::Event> *eventWaitList,
			cl::Event *startEvent,
			cl::Event *endEvent) = 0;

	/*!
	 * \brief Receive data from host.
	 * This is a non-blocking operation.
	 *
	 * \param[in]  size     size of data buffer
	 * \param[out] ptr      data buffer
	 * \return
	 */
	virtual std::shared_ptr<DataTransfer> receiveData(
			dcl::transfer_id transfer_id,
			size_t  size,
			void *  ptr,
			bool skip_compress_step = false,
			const std::shared_ptr<Completable> &trigger_event = nullptr) = 0;

	virtual void receiveDataToClBuffer(
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
			cl::Event *endEvent) = 0;
};

} /* namespace dcl */

#endif /* DCL_PROCESS_H_ */
