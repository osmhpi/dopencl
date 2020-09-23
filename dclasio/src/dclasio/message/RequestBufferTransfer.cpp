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
 * \file RequestBufferTransfer.cpp
 *
 * \date 2020-02-26
 * \author Joan Bruguera
 */

#include <dclasio/message/RequestBufferTransfer.h>
#include <dclasio/message/Request.h>

#include <dcl/DCLTypes.h>

#include <cstddef>

namespace dclasio {
namespace message {

RequestBufferTransfer::RequestBufferTransfer() {
}

RequestBufferTransfer::RequestBufferTransfer(
		dcl::object_id bufferId, dcl::transfer_id transferId) :
	_bufferId(bufferId), _transferId(transferId) {
}

RequestBufferTransfer::RequestBufferTransfer(const RequestBufferTransfer& rhs) :
	Request(rhs), _bufferId(rhs._bufferId), _transferId(rhs._transferId) {
}

dcl::object_id RequestBufferTransfer::bufferId() const {
	return _bufferId;
}

dcl::transfer_id RequestBufferTransfer::transferId() const {
	return _transferId;
}

} /* namespace message */
} /* namespace dclasio */
