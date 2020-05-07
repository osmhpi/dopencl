# dOpenCL

[![Build Status](https://travis-ci.org/joanbm/dopencl.svg?branch=master)](https://travis-ci.org/joanbm/dopencl)

# Introduction

dOpenCL allows programmers to access remote OpenCL devices that are connected to the host machine via network. As dOpenCL relies on API forwarding, the OpenCL code doesn't have to be changed in order to function properly. Instead dOpenCL handles all the network transfers.

dOpenCL has been created by a research group at the Uni Münster, Germany. Therefore all credit for the initial project goes to them. If you want to find out more about their awesome project, you can visit [their dOpenCL web page](http://www.uni-muenster.de/PVS/en/research/dopencl/) or read [their paper](http://ieeexplore.ieee.org/document/6270637/). Regarding the paper there are also sources that provide it for free.

This version of dOpenCL is heavily based on the C++ library Boost.Asio and has been abandoned by the research group as they switched to an internal solution for the network transfers, called ['The Real-Time Framework'](http://www.uni-muenster.de/PVS/en/research/rtf/index.html). Unfortunately, this dependency is not openly accessible, which is why this version was derived from the Boost.Asio-based version (particularly, from Uni Münsters's SVN r1819 available [here](https://github.com/joanbm/dopencl/tree/r1819)).

# Modifications introduced by this fork

This version of dOpenCL has been modified to function with Aparapi. Aparapi allows OpenCL programmers to write and execute their Kernels in pure Java, which allows much faster progression and especially aids beginners to execute their first OpenCL programs on the CPU and GPU without knowledge about internals. If you want to find out more about Aparapi visit their [GitHub repository](https://github.com/aparapi/aparapi). In order to connect Aparapi and dOpenCL with each other, [a specialized fork](https://github.com/joanbm/aparapi) (created by Florian Rösler) is required. It fixes some issues that arise when combining both libraries and enables to use several dynamic features like adding devices to the cluster at runtime. dOpenCL is also a fundamental component of the [CloudCL Framework](https://github.com/joanbm/cloudcl).

Additionally, this version of dOpenCL has been modified to add support for I/O Link Compression. I/O Link Compression is a technique for accelerating data transfers though the network by using extremely fast, hardware-accelerated compression techniques. Those are provided through the 842 compression algorithm by [lib842](https://github.com/joanbm/lib842).

Additionally, a lot of problems that were present in the original dOpenCL implementation this fork derived from have been fixed, so this version is substantially more stable.

# Further information

For further information, see the following files:

* `README_Linux.txt`: Basic information about dOpenCL

* `INSTALL_Linux.txt`: How to build dOpenCL

* `COPYING` and `COPYING.academic`: Distribution licenses
