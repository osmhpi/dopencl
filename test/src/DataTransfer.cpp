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
 * \file DataTransfer.cpp
 *
 * DataTransfer test suite
 *
 * \date 2020-02-13
 * \author Joan Bruguera
 */

#define BOOST_TEST_MODULE DataTransfer
#include <boost/test/unit_test.hpp>

#include "../../dclasio/src/dclasio/comm/ConnectionListener.h"
#include "../../dclasio/src/dclasio/comm/DataDispatcher.h"
#include "../../dclasio/src/dclasio/comm/DataStream.h"


/* ****************************************************************************
 * Test cases
 ******************************************************************************/

class test_connection_listener : public dclasio::comm::connection_listener {
public:
    // Blocks the thread until the DataStream has been successfully connected
    dclasio::comm::DataStream *wait_for_datastream_connected() {
        std::unique_lock<std::mutex> lock(mutex);
        while (connected_ds == nullptr)
            cv.wait(lock);
        return connected_ds;
    }

private:
    std::mutex mutex;
    std::condition_variable cv;
    dclasio::comm::DataStream *connected_ds = nullptr;

    bool approve_message_queue(
            dclasio::ProcessImpl::Type process_type,
            dcl::process_id process_id) override {
        return true;
    }

    void message_queue_connected(
            dclasio::comm::message_queue& msgq,
            dclasio::ProcessImpl::Type process_type,
            dcl::process_id process_id) override {
    }

    void message_queue_disconnected(
            dclasio::comm::message_queue& msgq) override {
    }

    bool approve_data_stream(
            dcl::process_id process_id) override {
        return true;
    }

    void data_stream_connected(
            dclasio::comm::DataStream& dataStream,
            dcl::process_id process_id) override {
        std::unique_lock<std::mutex> lock(mutex);
        connected_ds = &dataStream;
        cv.notify_one();
    }
};

void validate_data_is_successfully_transferred(const std::vector<std::vector<uint8_t>> &send_buffers) {
    // Set up two DataStreams connected to each other
    test_connection_listener listener1;
    test_connection_listener listener2;

    boost::asio::ip::tcp::endpoint ep1(boost::asio::ip::address_v4::loopback(), 22222);
    boost::asio::ip::tcp::endpoint ep2(boost::asio::ip::address_v4::loopback(), 55555);
    dcl::process_id process_id1 = { 1 ,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }; // UUID
    dcl::process_id process_id2 = { 2 ,2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2 }; // UUID

    dclasio::comm::DataDispatcher dd1(process_id1);
    dclasio::comm::DataDispatcher dd2(process_id2);

    dd1.add_connection_listener(listener1);
    dd2.add_connection_listener(listener2);

    dd1.bind(ep1);
    dd2.bind(ep2);

    dd1.start();
    dd2.start();

    dclasio::comm::DataStream *ds1 = dd1.create_data_stream(ep2);
    dclasio::comm::DataStream *ds2 = dd2.create_data_stream(ep1);

    ds1->connect(process_id1);
    ds2->connect(process_id2);

    auto connected_ds1 = listener1.wait_for_datastream_connected();
    auto connected_ds2 = listener2.wait_for_datastream_connected();

    // Send the data through one of the datastreams
    for (const auto &send_buffer : send_buffers)
        connected_ds1->write(send_buffer.size(), send_buffer.data());

    // Read the data through the other end
    std::vector<std::vector<uint8_t>> recv_buffers(send_buffers.size());
    for (size_t i = 0; i < send_buffers.size(); i++) {
        recv_buffers[i].resize(send_buffers[i].size(), 0xFF);
    }

    std::mutex mutex_data_available;
    std::condition_variable cv_data_available;
    size_t num_available = 0;

    for (auto &recv_buffer : recv_buffers) {
        auto recv_handle = ds2->read(recv_buffer.size(), recv_buffer.data());

        recv_handle->setCallback([&mutex_data_available, &cv_data_available, &num_available](cl_int notify) {
            std::unique_lock <std::mutex> lock(mutex_data_available);
            num_available++;
            cv_data_available.notify_one();
        });
    }

    std::unique_lock<std::mutex> lock(mutex_data_available);
    while (num_available < send_buffers.size()) {
        cv_data_available.wait(lock);
    }

    // Clean up
    dd1.destroy_data_stream(ds1);
    dd2.destroy_data_stream(ds2);
    dd2.stop();
    dd2.stop();

    for (size_t i = 0; i < send_buffers.size(); i++) {
        BOOST_CHECK_EQUAL_COLLECTIONS(recv_buffers[i].begin(), recv_buffers[i].end(),
                                      send_buffers[i].begin(), send_buffers[i].end());
    }
}

BOOST_AUTO_TEST_CASE( DataTransfer_Small )
{
    static const std::vector<uint8_t> send_buffer = { 0x13, 0x37, 0xB3, 0x3F, 0x13, 0x37, 0xB3, 0x3F };
    validate_data_is_successfully_transferred({send_buffer});
}

BOOST_AUTO_TEST_CASE( DataTransfer_Big_Compressible )
{
    std::vector<uint8_t> send_buffer(32 * 1024 * 1024);
    for (size_t i = 0; i < send_buffer.size(); i++) {
        send_buffer[i] = (uint8_t)(i / 1024);
    }

    validate_data_is_successfully_transferred({send_buffer});
}

static void random_fill(std::vector<uint8_t> &vec) {
    unsigned xorshift_seed = 123456;
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = (uint8_t)xorshift_seed;

        xorshift_seed ^= xorshift_seed << 13;
        xorshift_seed ^= xorshift_seed >> 17;
        xorshift_seed ^= xorshift_seed << 5;
    }
}

BOOST_AUTO_TEST_CASE( DataTransfer_Big_Uncompressible )
{
    std::vector<uint8_t> send_buffer(32 * 1024 * 1024);
    random_fill(send_buffer);

    validate_data_is_successfully_transferred({send_buffer});
}

BOOST_AUTO_TEST_CASE( DataTransfer_Multi_Uncompressible )
{
    std::vector<std::vector<uint8_t>> send_buffers(30, std::vector<uint8_t>(1024 * 100));
    for (auto &send_buffer : send_buffers)
        random_fill(send_buffer);
    validate_data_is_successfully_transferred(send_buffers);
}
