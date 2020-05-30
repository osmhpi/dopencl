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
 * \file DataTransferUtils.cpp
 *
 * \date 2020-05-30
 * \author Joan Bruguera
 */

#include <dcl/util/Logger.h>

#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION)
#include <lib842/hw.h>
#endif
#ifdef IO_LINK_COMPRESSION
#include <cstdlib>
#include <thread>
#endif

namespace dclasio {

namespace comm {

#ifdef IO_LINK_COMPRESSION
unsigned int determine_io_link_compression_num_threads(const char *env_name) {
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

bool determine_io_link_compression_spread_threads(const char *env_name) {
    return std::getenv(env_name) != nullptr;
}
#endif

#if defined(IO_LINK_COMPRESSION) && defined(USE_HW_IO_LINK_COMPRESSION) && defined(LIB842_HAVE_CRYPTODEV_LINUX_COMP)
bool is_hw_io_link_compression_enabled() {
    static bool enabled = std::getenv("DCL_DISABLE_HW_IO_LINK_COMPRESSION") == nullptr;

    if (!hw842_available()) {
        dcl::util::Logger << dcl::util::Info
                    << "Hardware 842 compression not available, falling back to software 842 compression."
                    << std::endl;
        return false;
    }

    return enabled;
}
#endif

} // namespace comm

} // namespace dclasio
