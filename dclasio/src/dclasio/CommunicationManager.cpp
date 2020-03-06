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
 * \file CommunicationManager.cpp
 *
 * \date 2011-10-26
 * \author Philipp Kegel
 */

#include "ComputeNodeCommunicationManagerImpl.h"
#include "HostCommunicationManagerImpl.h"

#include <dcl/CommunicationManager.h>
#include <dcl/DCLException.h>

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace {

boost::log::trivial::severity_level getSeverity() {
    const char *loglevel = getenv("DCL_LOG_LEVEL");

    if (loglevel) {
        if (strcmp(loglevel, "ERROR") == 0) {
            return boost::log::trivial::error;
        } else if (strcmp(loglevel, "WARNING") == 0) {
            return boost::log::trivial::warning;
        } else if (strcmp(loglevel, "INFO") == 0) {
            return boost::log::trivial::info;
        } else if (strcmp(loglevel, "DEBUG") == 0) {
            return boost::log::trivial::debug;
        } else if (strcmp(loglevel, "VERBOSE") == 0) {
            return boost::log::trivial::trace;
        }
    }

    // use default log level
#ifndef NDEBUG
    return boost::log::trivial::debug;
#else
    return boost::log::trivial::info;
#endif
}

} /* unnamed namespace */

/* ****************************************************************************/

namespace dcl {

HostCommunicationManager * HostCommunicationManager::create() {
    // set up dOpenCL logger
    boost::log::add_file_log
    (
        boost::log::keywords::file_name = "dcl_host.log",
        boost::log::keywords::format = "(%Severity%) [%TimeStamp%]: %Message%"
    );
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= getSeverity());
    boost::log::add_common_attributes();

    return new dclasio::HostCommunicationManagerImpl();
}

/* ****************************************************************************/

ComputeNodeCommunicationManager * ComputeNodeCommunicationManager::create(
        const std::string& url) {
    std::string host;
    dclasio::port_type port = dclasio::CommunicationManagerImpl::DEFAULT_PORT;
    dclasio::CommunicationManagerImpl::resolve_url(url, host, port);
    if (host.empty()) {
        throw dcl::InvalidArgument(DCL_INVALID_NODE, "Missing host name");
    }

    // generate name of dOpenCL log file
    std::string logFileName;
    {
        std::stringstream ss;

        /* TODO Log to system default location of files
         * Log to /var/log/dcld */
        ss << "dcl_" << host << ".log";
        ss >> logFileName;
    }
    // set up dOpenCL logger
    boost::log::add_file_log
    (
        boost::log::keywords::file_name = logFileName.c_str(),
        boost::log::keywords::format = "(%Severity%) [%TimeStamp%]: %Message%"
    );
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= getSeverity());
    boost::log::add_common_attributes();

    return new dclasio::ComputeNodeCommunicationManagerImpl(host, port);
}

} /* namespace dcl */
