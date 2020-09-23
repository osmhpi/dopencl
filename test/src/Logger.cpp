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
 * Logger test suite
 *
 * \date 2020-04-28
 * \author Joan Bruguera
 */

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define BOOST_TEST_MODULE Platform
#include <boost/test/unit_test.hpp>

#include <dcl/util/Logger.h>

#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <string>
#include <iterator>
#include <algorithm>
#include <iostream>

/* ****************************************************************************
 * Test cases
 ******************************************************************************/

BOOST_AUTO_TEST_CASE( MultithreadLoggerTest )
{
    // Arrange
    static constexpr unsigned NUM_THREADS = 4;
    static constexpr unsigned NUM_LINES = 40000;
    static const std::string LOG_PATH = "test.log";

    std::vector<std::string> expected;
    for (size_t i = 0; i < NUM_THREADS * NUM_LINES; i++)
        expected.push_back(std::to_string(i));

    // Act
    {
        static std::ofstream dclLogFileOut(LOG_PATH);
        dcl::util::Logger.setOutput(dclLogFileOut);
        dcl::util::Logger.setLoggingLevel(dcl::util::Severity::Info);

        std::atomic<int> counter{0};
        std::vector<std::thread> threads;
        for (size_t i = 0; i < NUM_THREADS; i++) {
            threads.emplace_back([&counter] {
                for (size_t j = 0; j < NUM_LINES; j++) {
                    dcl::util::Logger << dcl::util::Error << counter++ << std::endl;
                }
            });
        }

        for (auto &t : threads)
            t.join();

        dclLogFileOut.flush();
    }

    std::ifstream dclLogFileIn(LOG_PATH);
    std::vector<std::string> actual;
    for (std::string line; std::getline(dclLogFileIn, line); ) {
        size_t idx = line.find_last_of(" \t");
        actual.push_back(idx != std::string::npos ? line.substr(idx+1) : line);
    }

    // Assert
    std::sort(actual.begin(), actual.end());
    std::sort(expected.begin(), expected.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(actual.begin(), actual.end(),
                                  expected.begin(), expected.end());
}
