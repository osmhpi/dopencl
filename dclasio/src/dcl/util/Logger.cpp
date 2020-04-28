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
 * \file Logger.cpp
 *
 * \date 2011-08-08
 * \author Michel Steuwer <michel.steuwer@uni-muenster.de>
 * \author Philipp Kegel <philipp.kegel@uni-muenster.de>
 */

#include <dcl/util/Logger.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <ostream>

namespace {

auto start = std::chrono::high_resolution_clock::now();

} // unnamed namespace

/******************************************************************************/

namespace dcl {

namespace util {

// LoggerImpl implementation

LoggerImpl& LoggerImpl::get() {
    static LoggerImpl logger;
    return logger;
}

void LoggerImpl::setOutput(std::ostream& output) {
    _output = output;
}

/* TODO Distinguish logging level and message severity
 * Logging level may be set to 'NONE' to disable logging, but message severity
 * must not be 'NONE' */
void LoggerImpl::setLoggingLevel(Severity severity) {
    _maxSeverity = severity;
}

void LoggerImpl::log(Severity severity, const std::string& str) {
    if (severity <= _maxSeverity) {
        std::lock_guard<std::mutex> lock(_mutex);
        auto time = std::chrono::high_resolution_clock::now() - start;
        _output.get() << severityToString(severity) << " ["
                << std::chrono::duration_cast<std::chrono::seconds>(time).count()
                << ':' << std::setw(6) << std::setfill('0')
                << (std::chrono::duration_cast<std::chrono::microseconds>(time).count() % 1000000)
                << "] " << str;
        // flush output
        _output.get().flush();
    }
}

LoggerImpl::LoggerImpl() :
    _output(std::clog), _maxSeverity(Severity::Warning) {
}

std::string LoggerImpl::severityToString(Severity severity) {
    switch (severity) {
    case Severity::Error:   return "ERROR  ";
    case Severity::Warning: return "WARNING";
    case Severity::Info:    return "INFO   ";
    case Severity::Debug:   return "DEBUG  ";
    case Severity::Verbose: return "VERBOSE";
    default:                return "       ";
    }
}

LoggerImpl& Logger = LoggerImpl::get();

// LoggerStream implementation

LoggerStream::LoggerStream(LoggerImpl& logger) :
        std::ostream(&_buffer), _buffer(logger) {
}

void LoggerStream::setCurrentSeverity(Severity severity) {
    _buffer.setCurrentSeverity(severity);
}

// LoggerBuffer implementation

LoggerStream::LoggerBuffer::LoggerBuffer(LoggerImpl& logger) :
        std::stringbuf(), _logger(logger), _currentSeverity(Severity::Info) {
}

void LoggerStream::LoggerBuffer::setCurrentSeverity(Severity severity) {
    _currentSeverity = severity;
}

int LoggerStream::LoggerBuffer::sync() {
    _logger.log(_currentSeverity, str());
    // clear buffer content
    str("");
    return 0;
}

LoggerStream& operator<<(
        LoggerImpl& logger,
        LoggerStream &(*manipulator)(LoggerStream &)) {
    // Different threads should be able to simultaneously log. For this, each
    // thread gets an independent stream and only syncs at the end. Note that:
    // * Both stream (std::ostream) and buffer (std::stringbuf) are not thread-safe,
    //   so using thread locals at the stream level is the right choice
    // * Using a static thread_local works fine here, since the logger is a singleton
    //   (i.e. there can't multiple loggers, each with its own thread local streams)
    static thread_local LoggerStream stream(logger);
    return manipulator(stream);
}

// LoggerStream manipulators
LoggerStream& Error(LoggerStream& stream) {
    stream.setCurrentSeverity(Severity::Error);
    return stream;
}

LoggerStream& Warning(LoggerStream& stream) {
    stream.setCurrentSeverity(Severity::Warning);
    return stream;
}

LoggerStream& Info(LoggerStream& stream) {
    stream.setCurrentSeverity(Severity::Info);
    return stream;
}

LoggerStream& Debug(LoggerStream& stream) {
    stream.setCurrentSeverity(Severity::Debug);
    return stream;
}

LoggerStream& Verbose(LoggerStream& stream) {
    stream.setCurrentSeverity(Severity::Verbose);
    return stream;
}

} // namespace util

} // namespacce dcl

