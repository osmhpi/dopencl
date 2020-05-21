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
 * \file    DataTransfer.h
 *
 * \date    2011-12-17
 * \author  Philipp Kegel
 *
 * C++ API declarations for dOpenCL communication layer
 */

#ifndef DATATRANSFER_H_
#define DATATRANSFER_H_

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <dcl/Completable.h>
#include <functional>

#if defined(IO_LINK_COMPRESSION)
#include <cstdlib>
#endif
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION)
#include <lib842/cl.h>
#endif

namespace dcl {

/*!
 * \brief A handle for an asynchronous data transfer.
 */
class DataTransfer : public Completable {
public:
	virtual ~DataTransfer() { }

    virtual void setCallback(
            const std::function<void (cl_int)>& notify) = 0;

	virtual cl_ulong submit() const = 0;
	virtual cl_ulong start() const = 0;
	virtual cl_ulong end() const = 0;

    virtual bool isComplete() const = 0;

	/*!
	 * \brief Blocks until this data transfer is complete.
	 */
	virtual void wait() const = 0;

	/*!
	 * \brief Aborts this data transfer.
	 *
	 * The data transfer is considered as failed after calling this method.
	 * All registered callbacks are called accordingly.
	 */
	virtual void abort() = 0;
};

#ifdef IO_LINK_COMPRESSION
static bool is_io_link_compression_enabled() {
    static bool enabled = std::getenv("DCL_DISABLE_IO_LINK_COMPRESSION") == nullptr;
    return enabled;
}
#endif

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
static bool is_cl_io_link_compression_enabled() {
    static bool enabled = std::getenv("DCL_DISABLE_CL_IO_LINK_COMPRESSION") == nullptr;
    return enabled;
}

static bool is_cl_io_link_compression_mode_inplace() {
    static bool enabled = std::getenv("DCL_CL_IO_LINK_COMPRESSION_INPLACE") != nullptr;
    return enabled;
}
#endif

/**
 * Base class containing the context for data transfers from/to OpenCL buffers.
 */
class CLDataTransferContext {
protected:
    CLDataTransferContext(
        const cl::Context &context,
        const cl::CommandQueue &commandQueue)
        : _context(context),
          _commandQueue(commandQueue)
    { }

public:
    cl::Context context() const {
        return _context;
    }

    cl::CommandQueue commandQueue() const {
        return _commandQueue;
    }

private:
    cl::Context _context;
    cl::CommandQueue _commandQueue;
};

/**
 * Context for incoming (i.e. receive to buffer) data transfers to OpenCL buffers.
 */
class CLInDataTransferContext : public CLDataTransferContext {

public:
    CLInDataTransferContext(
        const cl::Context &context,
        const cl::CommandQueue &commandQueue
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
        , const lib842::CLDeviceDecompressor *cl842DeviceDecompressor
#endif
    )
        : CLDataTransferContext(context, commandQueue)
#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
          , _cl842DeviceDecompressor(cl842DeviceDecompressor)
#endif
    { }

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_OPENCL)
    const lib842::CLDeviceDecompressor *cl842DeviceDecompressor() const {
        return _cl842DeviceDecompressor;
    }

    static constexpr size_t NUM_BUFFERS = 2;
    std::array<cl::Buffer, NUM_BUFFERS> &cl842WorkBuffers() const {
        return _cl842WorkBuffers;
    }

private:
    const lib842::CLDeviceDecompressor *_cl842DeviceDecompressor;
    mutable std::array<cl::Buffer, NUM_BUFFERS> _cl842WorkBuffers;
#endif
};

/**
 * Context for outbound (i.e. send from buffer) data transfers from OpenCL buffers.
 */
class CLOutDataTransferContext : public CLDataTransferContext {
public:
    CLOutDataTransferContext(
        const cl::Context &context,
        const cl::CommandQueue &commandQueue)
        : CLDataTransferContext(context, commandQueue)
    { }
};

} /* namespace dcl */

#endif /* DATATRANSFER_H_ */
