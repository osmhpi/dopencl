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
 * \file ByteBuffer.h
 *
 * \date 2014-03-20
 * \author Philipp Kegel
 */

#ifndef DCL_BYTEBUFFER_H_
#define DCL_BYTEBUFFER_H_

#include <dcl/Binary.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <boost/uuid/uuid.hpp>

namespace {

/*!
 * \brief Serialization type traits
 */
template<typename T> struct serialization;
template<> struct serialization<cl_char>   { static const size_t size = sizeof(cl_char);   }; // 8 bit
template<> struct serialization<cl_uchar>  { static const size_t size = sizeof(cl_uchar);  }; // 8 bit
template<> struct serialization<cl_short>  { static const size_t size = sizeof(cl_short);  }; // 16 bit
template<> struct serialization<cl_ushort> { static const size_t size = sizeof(cl_ushort); }; // 16 bit
template<> struct serialization<cl_int>    { static const size_t size = sizeof(cl_int);    }; // 32 bit
template<> struct serialization<cl_uint>   { static const size_t size = sizeof(cl_uint);   }; // 32 bit
template<> struct serialization<cl_long>   { static const size_t size = sizeof(cl_long);   }; // 64 bit
template<> struct serialization<cl_ulong>  { static const size_t size = sizeof(cl_ulong);  }; // 64 bit
template<> struct serialization<float>     { static const size_t size = sizeof(float);     }; // 32 bit
template<> struct serialization<double>    { static const size_t size = sizeof(double);    }; // 64 bit
template<> struct serialization<boost::uuids::uuid> { static const size_t size = sizeof(boost::uuids::uuid);  }; // 128 bit

} // anonymous namespace

/* ****************************************************************************/

namespace dcl {

/*!
 * \brief A simple de-/serialization facility
 * This class is able to serialize the following types:
 *  + OpenCL API types (cl_int, cl_uint, cl_ulong, ...),
 *  + float, double,
 *  + size_t,
 *  + C strings, std::string, and
 *  + std::vector<T>, where T is any serializable type.
 * Network byte order is used for serialized representation to ensure portability.
 * Deserialization is *not* type-safe, i.e., it is the caller's responsibility
 * to extract serialized data correctly.
 * This class is not thread-safe for performance reasons.
 */
class OutputByteBuffer {
public:
    typedef char value_type;

    const static size_t DEFAULT_SIZE = 512; //!< default buffer size in bytes

private:
    /*!
     * \brief Ensures that at least \c size bytes can be written to the buffer
     * \param[in]  free the number of bytes to write
     */
    inline void ensure_free(
            size_t free) {
        auto size = _len + free;
        if (size > _bytes.size()) { // ensure required buffer size
            while (size < _bytes.size()) {
                size *= 2; // double buffer size
            }
            _bytes.resize(size);
        }
    }

public:
    OutputByteBuffer();
    /*!
     * \brief Creates a buffer with the specified number of reserved bytes
     * The buffer size as returned by OutputByteBuffer::size is 0.
     * \param[in]  initial_size the internal size of the byte buffer
     */
    OutputByteBuffer(
            size_t initial_size);

    template<typename T>
    OutputByteBuffer& operator<<(
            const T& value) {
        ensure_free(serialization<T>::size);
        // TODO Convert to network byte order
        // TODO Use std::copy
        memcpy(_bytes.data() + _len, &value, serialization<T>::size);
        _len += serialization<T>::size;
        return *this;
    }

    OutputByteBuffer& operator<<(
            const bool flag);
    OutputByteBuffer& operator<<(
            const std::string& str);
    OutputByteBuffer& operator<<(
            const Binary& data);

    /*!
     * \brief Serializes a vector of serializable values
     * \param[in]  values   the vector to serialize
     * \return this byte buffer
     */
    template<typename T>
    OutputByteBuffer& operator<<(
            const std::vector<T>& values) {
        operator<<(values.size()); // write number of elements
        for (const auto& value : values) {
            operator<<(value);
        }
        return *this;
    }

    /*!
     * \brief Serializes a map of serializable values
     * \param[in]  pairs    the map of key-value-pairs to serialize
     * \return this byte buffer
     */
    template<typename Key, typename Value>
    OutputByteBuffer& operator<<(
            const std::map<Key, Value>& pairs) {
        operator<<(pairs.size()); // write number of pairs
        for (const auto& pair : pairs) {
            operator<<(pair.first);
            operator<<(pair.second);
        }
        return *this;
    }

    size_t size() const;
    const value_type *data() const;

private:
    size_t _len; // actual valid buffer size
    std::vector<value_type> _bytes; // buffer data
};

class InputByteBuffer {
public:
    typedef char value_type;

    const static size_t DEFAULT_SIZE = 512; //!< default buffer size in bytes

private:
    /*!
     * \brief Ensures that at least \c size bytes can be read from the buffer
     * \param[in]  size the number of bytes to read
     * \throw std::out_of_range if less than \c size bytes can be read from the buffer
     */
    inline void ensure_bytes(size_t size) {
        if ((_len - _pos) < size) throw std::out_of_range("Buffer underflow");
    }

public:
    InputByteBuffer();
    /*!
     * \brief Creates a buffer with the specified number of reserved bytes
     * The buffer size as returned by InputByteBuffer::size is 0.
     * \param[in]  initial_size the internal size of the byte buffer
     */
    InputByteBuffer(size_t initial_size);

    template<typename T>
    InputByteBuffer& operator>>(
            T& value) {
        ensure_bytes(serialization<T>::size);
        // TODO Convert to host byte order
        // TODO Use std::copy
        memcpy(&value, _bytes.data() + _pos, serialization<T>::size);
        _pos += serialization<T>::size;
        return *this;
    }

    InputByteBuffer& operator>>(
            bool& flag);
    InputByteBuffer& operator>>(
            std::string& str);
    InputByteBuffer& operator>>(
            Binary& data);

    template<typename T>
    InputByteBuffer& operator>>(
            std::vector<T>& values) {
        size_t size;
        operator>>(size); // read number of elements
        values.resize(size);
        for (auto& value : values) {
            // remove const qualifier from value to update it from byte buffer
            operator>>(const_cast<typename std::remove_const<decltype(value)>::type>(value));
        }
        return *this;
    }

    template<typename Key, typename Value>
    InputByteBuffer& operator>>(
            std::map<Key, Value>& pairs) {
        size_t size;
        operator>>(size); // read number of pairs
        pairs.clear();
        for (size_t i = 0; i < size; ++i) {
            // remove const qualifiers from key and value to update them from byte buffer
            typename std::remove_const<Key>::type key;
            typename std::remove_const<Value>::type value;
            operator>>(key);
            operator>>(value);
            pairs.insert(std::make_pair(std::move(key), std::move(value)));
            // TODO std::map::emplace does not compile with GCC 4.6
//            pairs.emplace(std::move(key), std::move(value));
        }
        return *this;
    }

    /*!
     * \brief Resizes the buffer to the specified size
     * The buffer's content is undefined after this operation.
     * Usually, this method is used before overwriting the buffer directly using an iterator.
     * \param[in]  size the new buffer size
     */
    void resize(size_t size);

    size_t size() const;
    value_type *data();

private:
    size_t _pos; // read count
    size_t _len; // actual valid buffer size
    std::vector<value_type> _bytes; // buffer data
};

} // namespace dcl

#endif /* DCL_BYTEBUFFER_H_ */
