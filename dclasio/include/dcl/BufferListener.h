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
 * \file BufferListener.h
 *
 * \date 2020-02-26
 * \author Joan Bruguera
 */

#ifndef DCL_BUFFERLISTENER_H_
#define DCL_BUFFERLISTENER_H_

#include "DCLTypes.h"

#include <cstddef>

namespace dcl {

class Process;

/*!
 * \brief Remote buffer listener API
 *
 * A buffer listener is informed when a buffer needs to be transferred from the host to a compute node.
 */
class BufferListener {
public:
    virtual ~BufferListener() { }

    /*!
     * \brief Notifies a listener about a buffer transfer.
     */
    virtual void onRequestBufferTransfer(dcl::Process &process, dcl::transfer_id transferId) = 0;
};

} /* namespace dcl */

#endif /* DCL_BUFFERLISTENER_H_ */
