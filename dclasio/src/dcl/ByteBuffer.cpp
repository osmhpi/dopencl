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
 * \file ByteBuffer.cpp
 *
 * \date 2014-03-20
 * \author Philipp Kegel
 */

#include <dcl/Binary.h>
#include <dcl/ByteBuffer.h>

#include <algorithm>
#include <string>

namespace dcl {

OutputByteBuffer::OutputByteBuffer() :
    _len(0), _bytes(DEFAULT_SIZE) { }
OutputByteBuffer::OutputByteBuffer(size_t initial_size) :
    _len(0), _bytes(initial_size) { }

OutputByteBuffer& OutputByteBuffer::operator<<(const bool flag) {
    ensure_free(1);
    _bytes[_len] = (flag ? 1 : 0);
    ++_len;
    return *this;
}

OutputByteBuffer& OutputByteBuffer::operator<<(const std::string& str) {
    auto size = str.size();
    operator<<(size); // write number of characters
    ensure_free(size);
    std::copy(std::begin(str), std::end(str), _bytes.begin() + _len);
    _len += size;
    return *this;
}

OutputByteBuffer& OutputByteBuffer::operator<<(const Binary& data) {
    auto size = data.size();
    operator<<(size); // write number of bytes
    ensure_free(size);
    auto begin = static_cast<const value_type *>(data.value());
    std::copy(begin, begin + size, _bytes.begin() + _len);
    _len += size;
    return *this;
}

size_t OutputByteBuffer::size() const {
    return _len;
}

const OutputByteBuffer::value_type *OutputByteBuffer::data() const {
    return _bytes.data();
}

InputByteBuffer::InputByteBuffer() :
        _pos(0), _len(0), _bytes(DEFAULT_SIZE) { }
InputByteBuffer::InputByteBuffer(size_t initial_size) :
        _pos(0), _len(0), _bytes(initial_size) { }

InputByteBuffer& InputByteBuffer::operator>>(bool& flag) {
    ensure_bytes(1);
    flag = (_bytes[_pos] != 0);
    ++_pos;
    return *this;
}

InputByteBuffer& InputByteBuffer::operator>>(std::string& str) {
    size_t size;
    operator>>(size); // read number of characters
    ensure_bytes(size);
    str.assign(_bytes.data() + _pos, size);
    _pos += size;
    return *this;
}

InputByteBuffer& InputByteBuffer::operator>>(Binary& data) {
    size_t size;
    operator>>(size); // read number of bytes
    ensure_bytes(size);
    data.assign(size, _bytes.data() + _pos);
    _pos += size;
    return *this;
}

void InputByteBuffer::resize(size_t size_) {
    _bytes.resize(size_);
    _pos = 0;
    _len = size_;
}

size_t InputByteBuffer::size() const {
    return _len - _pos;
}

InputByteBuffer::value_type *InputByteBuffer::data() {
    return _bytes.data() + _pos;
}

} // namespace dcl
