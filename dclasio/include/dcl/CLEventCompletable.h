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
 * \file    CLEventCompletable.h
 *
 * \date    2020-03-01
 * \author  Joan Bruguera
 *
 * Allows wrapping an OpenCL event into a dcl::Completable.
 */

#ifndef CLEVENTCOMPLETABLE_H_
#define CLEVENTCOMPLETABLE_H_

#include "Completable.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <memory>

namespace dcl {

class CLEventCompletable : public Completable {
public:
    CLEventCompletable(cl::Event event) :
            _event(event) {
    }

    virtual void setCallback(const std::function<void(cl_int)> &notify) {
        // We need to dynamically allocate a copy the function and pass it through user_data
        // in order to keep the lambda captures through the C-style callback OpenCL offers
        auto notify_ptr = new std::function<void(cl_int)>(notify);

        _event.setCallback(CL_COMPLETE, [](cl_event, cl_int status, void *user_data) {
            std::unique_ptr<std::function<void(cl_int)>> notify_ptr(
                    static_cast<std::function<void(cl_int)> *>(user_data));
            (*notify_ptr)(status);
        }, notify_ptr);
    }

private:
    cl::Event _event;
};

}

#endif /* CLEVENTCOMPLETABLE_H_ */
