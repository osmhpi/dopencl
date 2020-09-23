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
 * \file GetDeviceIDs.h
 *
 * \date 2014-04-05
 * \author Philipp Kegel
 */

#ifndef GETDEVICEIDS_H_
#define GETDEVICEIDS_H_

#include <dclasio/message/Request.h>

#include <dcl/ByteBuffer.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace dclasio {
namespace message {

class GetDeviceIDs: public Request {
public:
    GetDeviceIDs() :
            deviceType(CL_DEVICE_TYPE_ALL) {
    }
    GetDeviceIDs(cl_device_type deviceType_) :
            deviceType(deviceType_) {
    }
    GetDeviceIDs(const GetDeviceIDs& rhs) :
            Request(rhs), deviceType(rhs.deviceType) {
    }
    virtual ~GetDeviceIDs() {
    }

    cl_device_type deviceType;

    static const class_type TYPE = 100 + GET_DEVICE_IDS;

    class_type get_type() const {
        return TYPE;
    }

    void pack(dcl::OutputByteBuffer& buf) const {
        Request::pack(buf);
        buf << deviceType;
    }

    void unpack(dcl::InputByteBuffer& buf) {
        Request::unpack(buf);
        buf >> deviceType;
    }
};

} /* namespace message */
} /* namespace dclasio */

#endif /* GETDEVICEIDS_H_ */
