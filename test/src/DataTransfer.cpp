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
#include <algorithm>
#include <fstream>
#include <vector>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <iostream>

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

/*!
 * \brief Sets up two pairs of remotely connected DataStreams, which simulates
 *        two different dOpenCL processes communicating with one another.
 */
class DataStreamPair {
public:
    test_connection_listener listener1;
    test_connection_listener listener2;
    boost::asio::ip::tcp::endpoint ep1;
    boost::asio::ip::tcp::endpoint ep2;
    dcl::process_id process_id1;
    dcl::process_id process_id2;
    dclasio::comm::DataDispatcher dd1;
    dclasio::comm::DataDispatcher dd2;
    dclasio::comm::DataStream *ds1;
    dclasio::comm::DataStream *ds2;
    dclasio::comm::DataStream *connected_ds1;
    dclasio::comm::DataStream *connected_ds2;

    DataStreamPair() :
        listener1(),
        listener2(),
        ep1(boost::asio::ip::address_v4::loopback(), 22222),
        ep2(boost::asio::ip::address_v4::loopback(), 55555),
        dd1(process_id1),
        dd2(process_id2) {
        // Set up two DataStreams connected to each other
        dd1.add_connection_listener(listener1);
        dd2.add_connection_listener(listener2);

        dd1.bind(ep1);
        dd2.bind(ep2);

        dd1.start();
        dd2.start();

        ds1 = dd1.create_data_stream(ep2);
        ds2 = dd2.create_data_stream(ep1);

        ds1->connect(process_id1);
        ds2->connect(process_id2);

        connected_ds1 = listener1.wait_for_datastream_connected();
        connected_ds2 = listener2.wait_for_datastream_connected();
    }

    ~DataStreamPair() {
        dd1.destroy_data_stream(ds1);
        dd2.destroy_data_stream(ds2);
        dd2.stop();
        dd2.stop();
    }
};


/*!
 * \brief Transfers the specified list of buffers from one DataStream to another,
 *        and validates that the data received from the other end is correct.
 */
void validate_data_is_successfully_transferred(const std::vector<std::vector<uint8_t>> &send_buffers) {
    DataStreamPair dsp;

    std::vector<std::vector<uint8_t>> recv_buffers(send_buffers.size());
    for (size_t i = 0; i < send_buffers.size(); i++) {
        recv_buffers[i].resize(send_buffers[i].size(), 0xFF);
    }

    // Send the data through one of the datastreams
    for (const auto &send_buffer : send_buffers)
        dsp.connected_ds1->write(send_buffer.size(), send_buffer.data(), false, nullptr);

    // Read the data through the other end
    std::vector<std::shared_ptr<dclasio::comm::DataReceipt>> recv_handles;
    for (auto &recv_buffer : recv_buffers)
        recv_handles.push_back(dsp.ds2->read(recv_buffer.size(), recv_buffer.data(), false, nullptr));
    for (auto &recv_handle : recv_handles)
        recv_handle->wait();

    // Assert that the received data is equal to the sent data
    for (size_t i = 0; i < send_buffers.size(); i++) {
        BOOST_CHECK_EQUAL_COLLECTIONS(recv_buffers[i].begin(), recv_buffers[i].end(),
                                      send_buffers[i].begin(), send_buffers[i].end());
    }
}

/*!
 * \brief Fills the range [first, last) with repetitive data,
 *        This should make the data compressible by the I/O compression algorithm.
 */
template<class iterator_type>
static void pattern_fill(iterator_type first, iterator_type last) {
    unsigned i = 0;
    for (iterator_type it = first; it != last; it++) {
        *it = (uint8_t)(i / 1024);
        i++;
    }
}

/*!
 * \brief Fills the range [first, last) with pseudo-random data,
 *        This should make the data non-compressible by the I/O compression algorithm.
 */
template<class iterator_type>
static void random_fill(iterator_type first, iterator_type last) {
    unsigned xorshift_seed = 123456;
    for (iterator_type it = first; it != last; it++) {
        *it = (uint8_t)xorshift_seed;

        xorshift_seed ^= xorshift_seed << 13;
        xorshift_seed ^= xorshift_seed >> 17;
        xorshift_seed ^= xorshift_seed << 5;
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
    pattern_fill(send_buffer.begin(), send_buffer.end());
    validate_data_is_successfully_transferred({send_buffer});
}

BOOST_AUTO_TEST_CASE( DataTransfer_Big_Uncompressible )
{
    std::vector<uint8_t> send_buffer(32 * 1024 * 1024);
    random_fill(send_buffer.begin(), send_buffer.end());
    validate_data_is_successfully_transferred({send_buffer});
}

BOOST_AUTO_TEST_CASE( DataTransfer_Multi_Uncompressible )
{
    std::vector<std::vector<uint8_t>> send_buffers(30, std::vector<uint8_t>(1024 * 100));
    for (auto &send_buffer : send_buffers)
        random_fill(send_buffer.begin(), send_buffer.end());
    validate_data_is_successfully_transferred(send_buffers);
}

BOOST_AUTO_TEST_CASE( DataTransfer_ProxyCompressedData )
{
    DataStreamPair dsp;

    std::vector<uint8_t> send_buffer(32 * 1024 * 1024);
    pattern_fill(send_buffer.begin(), send_buffer.begin() + send_buffer.size()/2);
    random_fill(send_buffer.begin() + send_buffer.size()/2, send_buffer.end());

    // Send the data through one of the datastreams
    dsp.connected_ds1->write(send_buffer.size(), send_buffer.data(), false, nullptr);

    // Read the data through the other end, but request that no decompression happens
    // (set skip_compress_step=true). This means that, if I/O link compression is enabled,
    // intermediate_buffer will contain the data in a compressed form.
    std::vector<uint8_t> intermediate_buffer(send_buffer.size(), 0xff);
    dsp.ds2->read(intermediate_buffer.size(), intermediate_buffer.data(), true, nullptr)->wait();

    bool intermediate_buffer_equal = std::equal(intermediate_buffer.begin(), intermediate_buffer.end(),
                                                send_buffer.begin());
    #ifdef IO_LINK_COMPRESSION
    // If I/O link compression is enabled, intermediate_buffer will contain the data
    // in a compressed form, so it will NOT be equal to the source data
    BOOST_CHECK(!intermediate_buffer_equal);
    #else
    // Otherwise, the data is just sent as usual (skip_compress_step=true is a no-op)
    BOOST_CHECK(intermediate_buffer_equal);
    #endif

    // Now test the opposite case, where we write the compressed data again
    // and request that it it not re-compressed, but read it as usual
    // This will then recover the original buffer we originally sent through the stream
    dsp.connected_ds2->write(intermediate_buffer.size(), intermediate_buffer.data(), true, nullptr);
    std::vector<uint8_t> recv_buffer(send_buffer.size());
    dsp.ds1->read(recv_buffer.size(), recv_buffer.data(), false, nullptr)->wait();

    BOOST_CHECK_EQUAL_COLLECTIONS(send_buffer.begin(), send_buffer.end(),
                                  recv_buffer.begin(), recv_buffer.end());
}

// ----------
// BENCHMARKS
// ----------

static constexpr const char *BENCHMARK_DATASET_PATH_ENV = "DATATRANSFER_BENCHMARK_DATASET_PATH";

static std::vector<uint8_t> load_file_to_vector(const char *file_name) {
    std::ifstream file(file_name, std::ifstream::ate | std::ifstream::binary);
    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    assert(file.tellg() < SIZE_MAX);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0);

    std::vector<uint8_t> file_data(size);
    file.read(reinterpret_cast<char *>(file_data.data()), file_data.size());
    return file_data;
}

static std::vector<uint8_t> load_benchmark_dataset() {
    const char *file_name = std::getenv(BENCHMARK_DATASET_PATH_ENV);
    if (file_name == nullptr) {
        std::cout
            << "'" << BENCHMARK_DATASET_PATH_ENV
            << "' environment variable not set, skipping test.\n";
        return std::vector<uint8_t>();
    }

    return load_file_to_vector(file_name);
}

class DataTransferBenchmarkClock {
    const char *test_name;
    size_t data_amount;
    std::chrono::time_point<std::chrono::steady_clock> start_time;

public:
    DataTransferBenchmarkClock(const char *test_name, size_t data_amount)
        : test_name(test_name), data_amount(data_amount),
          start_time(std::chrono::steady_clock::now())
    {}

    ~DataTransferBenchmarkClock() {
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

        auto data_amount_mb = static_cast<double>(data_amount) / 1048576.0;
        auto duration_s = static_cast<double>(duration_ms) / 1000.0;
        auto bandwidth_mbps = data_amount_mb / duration_s;

        std::cout
            << test_name << "\n"
            << "---\n"
            << "Duration: " << duration_ms << " ms\n"
            << "Bandwidth: " << bandwidth_mbps << " MB/s\n"
            << "\n";
    }
};

BOOST_AUTO_TEST_CASE( DataTransfer_Benchmark )
{
    auto dataset = load_benchmark_dataset();
    if (dataset.empty())
        return;
    std::vector<uint8_t> dataset_compressed(dataset.size(), 0xFF);

    DataStreamPair dsp;

    {
        DataTransferBenchmarkClock clock("COMPRESS", dataset.size());
        // Send uncompressed, receive compressed
        dsp.connected_ds1->write(dataset.size(), dataset.data(), false, nullptr);
        dsp.ds2->read(dataset_compressed.size(), dataset_compressed.data(), true, nullptr)->wait();
    }

    std::vector<uint8_t> dataset_recompressed(dataset.size(), 0xFF);
    {
        DataTransferBenchmarkClock clock("PROXY", dataset_compressed.size());
        // Send compressed, receive compressed
        dsp.connected_ds1->write(dataset_compressed.size(), dataset_compressed.data(), true, nullptr);
        dsp.ds2->read(dataset_recompressed.size(), dataset_recompressed.data(), true, nullptr)->wait();
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(dataset_compressed.begin(), dataset_compressed.end(),
                                  dataset_recompressed.begin(), dataset_recompressed.end());

    std::vector<uint8_t> dataset_uncompressed(dataset.size(), 0xFF);
    {
        DataTransferBenchmarkClock clock("DECOMPRESS", dataset.size());
        // Send compressed, receive uncompressed
        dsp.connected_ds1->write(dataset_compressed.size(), dataset_compressed.data(), true, nullptr);
        dsp.ds2->read(dataset_uncompressed.size(), dataset_uncompressed.data(), false, nullptr)->wait();
    }

    BOOST_CHECK_EQUAL_COLLECTIONS(dataset.begin(), dataset.end(),
                                  dataset_uncompressed.begin(), dataset_uncompressed.end());
}
