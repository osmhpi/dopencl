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
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <dcl/Completable.h>
#include <functional>
#include <cstdint>

#if defined(IO_LINK_COMPRESSION) && defined(USE_CL_IO_LINK_COMPRESSION)
#include <lib842/cl.h>
#endif
#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
#include <lib842/hw.h>
#endif
#if defined(IO_LINK_COMPRESSION)
#include <cstdlib>
#include <thread>
#include <dcl/util/Logger.h>
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

#ifdef IO_LINK_COMPRESSION
    // TODOXXX can this be moved to a better place?
    static constexpr size_t CL_UPLOAD_BLOCK_SIZE = static_cast<size_t>(1) << 29; // 512 MiB
#endif
};

} /* namespace dcl */

// TODOXXX: Move this to a better place

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
#endif

#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
static bool is_hw_io_link_compression_enabled() {
    static bool enabled = std::getenv("DCL_DISABLE_HW_IO_LINK_COMPRESSION") == nullptr;
    return enabled;
}
#endif

#ifdef IO_LINK_COMPRESSION
static unsigned int determine_io_link_compression_num_threads(const char *env_name) {
    // Configuration for the number of threads to use for compression or decompression
    const char *env_value = std::getenv(env_name);
    if (env_value != nullptr && std::atoi(env_value) > 0) {
        return (unsigned int)std::atoi(env_value);
    }

    // If the value is not specified (or invalid),
    // the hardware concurrency level (~= number of logical cores) is used
    static unsigned int hardware_concurrency = std::thread::hardware_concurrency();
    if (hardware_concurrency == 0) {
        dcl::util::Logger << dcl::util::Warning << __func__ << ": "
                          << "std::thread::hardware_concurrency() returned 0, using 1 thread"
                          << std::endl;
        return 1;
    }

    return hardware_concurrency;
}

static bool determine_io_link_compression_spread_threads(const char *env_name) {
    return std::getenv(env_name) != nullptr;
}
#endif

#endif /* DATATRANSFER_H_ */
