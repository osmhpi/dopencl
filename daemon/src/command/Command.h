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
 * \file Command.h
 *
 * \date 2012-03-23
 * \author Philipp Kegel
 */

#ifndef COMMAND_H_
#define COMMAND_H_

#include <dcl/DCLTypes.h>
#include <dcl/Host.h>

#ifdef __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

namespace dcld {

namespace command {

class Command {
public:
    Command(
            dcl::Host&              host,
            dcl::object_id          commandId,
            const cl::UserEvent&    event) :
            _host(host), _commandId(commandId), _event(event) {
    }

    virtual ~Command() { }

    /*!
     * \brief Executes this command.
     *
     * \param[in]  errcode  an error code of the operation that triggered this
     *             command's execution.
     */
    virtual void execute(
            cl_int errcode) = 0;

protected:
    dcl::Host& _host;
    dcl::object_id _commandId;
    cl::UserEvent _event;
};

} /* namespace command */

} /* namespace dcld */

#endif /* COMMAND_H_ */
