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
 * \file Logger.h
 *
 * \date 2011-08-08
 * \author Michel Steuwer <michel.steuwer@uni-muenster.de>
 * \author Philipp Kegel <philipp.kegel@uni-muenster.de>
 */

#ifndef LOGGER_H_
#define LOGGER_H_

#include <mutex>
#include <functional>
#include <ostream>
#include <sstream>

namespace dcl {

namespace util {

enum class Severity {
    Error = 1,
    Warning = 2,
    Info = 3,
    Debug = 4,
    Verbose = 5
};

/*!
 * \brief A simple logger
 */
class LoggerImpl {
public:
    static LoggerImpl& get();

    void setOutput(std::ostream& output);

    void setLoggingLevel(Severity severity);

    void log(Severity severity, const std::string& str);

private:
    LoggerImpl();
    static std::string severityToString(Severity severity);

    std::reference_wrapper<std::ostream> _output;
    Severity _maxSeverity;
    std::mutex _mutex; //!< Mutex to synchronize logging
};

extern LoggerImpl& Logger;

class LoggerStream: public std::ostream {
    explicit LoggerStream(LoggerImpl& logger);

    void setCurrentSeverity(Severity severity);

    // Only let this class be constructed and configured through the manipulators
    friend LoggerStream& operator<<(
        LoggerImpl& logger,
        LoggerStream &(*manipulator)(LoggerStream &));
    friend LoggerStream& Error(LoggerStream& stream);
    friend LoggerStream& Warning(LoggerStream& stream);
    friend LoggerStream& Info(LoggerStream& stream);
    friend LoggerStream& Debug(LoggerStream& stream);
    friend LoggerStream& Verbose(LoggerStream& stream);

    class LoggerBuffer: public std::stringbuf {
    public:
        explicit LoggerBuffer(LoggerImpl& logger);

        void setCurrentSeverity(Severity severity);

        int sync() override;
    private:
        LoggerImpl& _logger;
        Severity _currentSeverity;
    };

    LoggerBuffer _buffer;
};

/*
 * custom manipulators
 */

LoggerStream& operator<<(
        LoggerImpl& logger,
        LoggerStream &(*manipulator)(LoggerStream &));

LoggerStream& Error(LoggerStream& stream);

LoggerStream& Warning(LoggerStream& stream);

LoggerStream& Info(LoggerStream& stream);

LoggerStream& Debug(LoggerStream& stream);

LoggerStream& Verbose(LoggerStream& stream);

} // namespace util

} // namespace dcl

#endif // LOGGER_H_
