FROM ubuntu:bionic

# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        git \
        cmake \
        libboost-all-dev \
        opencl-headers \
        ocl-icd-opencl-dev \
        ca-certificates

# dOpenCL
RUN git clone https://github.com/osmhpi/dopencl /tmp/dopencl --recursive \
	&& mkdir /tmp/dopencl/build \
	&& cd /tmp/dopencl/build \
	&& cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS=-w .. \
	&& make \ 
	&& make install \
	&& cp -r /tmp/dopencl/dclasio/include/* /usr/local/include/ \
	&& rm -Rf /tmp/dopencl

ENV LD_LIBRARY_PATH="/usr/local/lib/:${LD_LIBRARY_PATH}"