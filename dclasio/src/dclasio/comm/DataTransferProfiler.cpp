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
 * \file DataTransferProfiler.cpp
 *
 * \date 2020-05-10
 * \author Joan Bruguera
 */

// Comment/uncomment this macro to enable/disables data transfer profiling
#define PROFILE_SEND_RECEIVE_BUFFER

#include "DataTransferProfiler.h"

#ifdef PROFILE_SEND_RECEIVE_BUFFER
#include <dcl/util/Logger.h>

#include <boost/uuid/uuid_io.hpp>

#include <chrono>
#endif

namespace dclasio {

namespace comm {

void profile_transfer(profile_transfer_direction direction,
                      dcl::transfer_id id, size_t size,
                      cl::Event &startEvent, cl::Event &endEvent) {
#ifdef PROFILE_SEND_RECEIVE_BUFFER
    struct profile_send_receive_buffer_times {
        profile_transfer_direction direction;
        dcl::transfer_id id;
        size_t size;
        std::chrono::time_point<std::chrono::steady_clock> enqueue_time;
        std::chrono::time_point<std::chrono::steady_clock> start_time;
        std::chrono::time_point<std::chrono::steady_clock> end_time;

        const char *direction_str() const {
            switch (direction) {
                case profile_transfer_direction::receive: return "Receive";
                case profile_transfer_direction::send:    return "Send";
                default:                                  return "???";
            }
        }
    };

    auto profile_times = new profile_send_receive_buffer_times();
    try {
        profile_times->direction = direction;
        profile_times->id = id;
        profile_times->size = size;
        profile_times->enqueue_time = std::chrono::steady_clock::now();

        startEvent.setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
            auto profile_times = ((profile_send_receive_buffer_times *)user_data);
            profile_times->start_time = std::chrono::steady_clock::now();
            dcl::util::Logger << dcl::util::Debug
                              << "(PROFILE) " << profile_times->direction_str() << " with id " << profile_times->id << " of size " << profile_times->size
                              << " started (ENQUEUE -> START) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                profile_times->start_time - profile_times->enqueue_time).count() << std::endl;
        }, profile_times);

        endEvent.setCallback(CL_COMPLETE, [](cl_event,cl_int,void *user_data) {
            std::unique_ptr<profile_send_receive_buffer_times> profile_times(
                static_cast<profile_send_receive_buffer_times *>(user_data));
            profile_times->end_time = std::chrono::steady_clock::now();
            dcl::util::Logger << dcl::util::Debug
                              << "(PROFILE) " << profile_times->direction_str() << " with id " << profile_times->id << " of size " << profile_times->size
                              << " uploaded (START -> END) on " << std::chrono::duration_cast<std::chrono::milliseconds>(
                profile_times->end_time - profile_times->start_time).count() << std::endl;
        }, profile_times);
    } catch (...) {
        delete profile_times;
        throw;
    }
#endif
}

} // namespace comm

} // namespace dclasio
