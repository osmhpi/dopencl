dOpenCL
=======

(last update: 2014-12-22, Philipp Kegel)

dOpenCL (distributed Open Compute Language) is an implementation of the OpenCL
API for distributed systems created by Philipp Kegel with the help of Michel
Steuwer, Sergei Gorlatch, and several students.

References:
* Kegel, P., Steuwer, M., Gorlatch, S., 2012. dOpenCL: Towards a Unified
  Programming Approach for Distributed Heterogeneous Multi-/Many-Core Systems.


--------
Contents
--------

1. System requirements
2. Getting started
3. Project structure
4. Known issues
5. Contact


-------------------
System requirements
-------------------

See 'INSTALL_Linux.txt' for system requirements and installation instructions.

dOpenCL has been successfully tested on the following systems:

* Ubuntu 12.04
* CentOS 5.6, 5.9
* Scientific Linux 5.6


---------------
Getting started
---------------

Before running the dOpenCL daemon or your OpenCL application, you have to ensure
that dOpenCL is properly installed (see 'INSTALL_Linux.txt').

Setting up your environment (optional)
--------------------------------------

Unless dOpenCL has been installed into your default system directories, you have
you have to update your environment's PATH and LD_LIBRARY_PATH as follows (note
the use of the DCL_HOME environment variable):

   export PATH=$PATH:$DCL_HOME/daemon
   export LD_LIBRARY_PATH=$DCL_HOME/dclgcf:$DCL_HOME/icdpp:$LD_LIBRARY_PATH

Starting and stopping the dOpenCL daemon
----------------------------------------

In order to start the dOpenCL daemon you need to have an OpenCL 1.1+ compliant
OpenCL implementation installed on your system.

The dOpenCL daemon has to be started on all nodes that should be accessible via
dOpenCL. Currently, the daemon does not become a daemon process but blocks the
terminal.

  dcld [-p<platform name>] <hostname>[:<port>]

Hint: Ubuntu resolves all hostnames of the local system to the address of the
loopback interface. In order to bind the dOpenCL daemon to another interface,
the interface's IP address has to be provided rather than the interface's host
name.

The -p option selects the native OpenCL platform that should be used by the
dOpenCL daemon. The specified platform name must be part of the platform's full
name. If no platform is specified, the system's OpenCL platform is used.

The daemon is stopped by sending it a SIGINT (press Strg+C) or SIGTERM (kill)
signal.

Running an OpenCL application with dOpenCL
------------------------------------------

As dOpenCL *is* an OpenCL implementation, existing applications do not have to
be compiled or linked anew in order to use dOpenCL. However, the following
preparations are required to connect the host and daemons an to make the
application use the dOpenCL ICD.

1. Create a node file

   Create a file named 'dcl.nodes' *in your application's working directory.*
   Add a line with a host name and optional port number (in the format
   <hostname>[:<port>]) for each node of your network, on which a dOpenCL
   daemon is running. For example:

     echo localhost >> dcl.nodes

   You can also create a global node file anywhere in your systems and export
   the DCL_NODE_FILE environment variable. If DCL_NODE_FILE is defined and not
   empty, its value is taken as the location of the application's node file.
   The 'dcl.nodes' file in the application's working directory will be ignored
   in this case.

2. Run the application

   If you have replaced the system's ICD by dOpenCL's ICD, you may run your
   application as usual.

   Otherwise, you have to use the LD_PRELOAD environment variable in order to
   preload the dOpenCL ICD before running your OpenCL application:

     LD_PRELOAD=libdOpenCL.so <application binary> [<arguments>]

   To avoid explicitly setting this variable each time you run your application,
   you may export LD_PRELOAD to your environment:

     export LD_PRELOAD="$LD_PRELOAD libdOpenCL.so"

   We recommend to create a start script for you application which you use to
   export LD_PRELOAD (and LD_LIBRARY_PATH as described in 'Setting up your
   environment') before starting you application.

Controlling log output
----------------------

dOpenCL creates log files for debugging purposes. In the daemon's working
directory, a file named 'dcl_<host name>.log' is created, where <host name> will
be the host name you selected when starting the damon. In the application's
working directory, a file 'dcl_host.log' will be created.

Currently, logging cannot be switched off, but the amount of log messages can be
controlled by setting the log level in the DCL_LOG_LEVEL environment variable.
The following settings are eligible:

  ERROR    only log error messages
  WARNING  log warnings
  INFO     log info messages  (default for release build)
  DEBUG    log debug messages (default for debug build)
  VERBOSE  log everything

If no log level is specified, the default log level is selected.

Note that the log files are deleted each time the daemon or application is
restarted.

If the DCL_LOG_TO_CONSOLE environment variable is set, the log is output
to the console, instead of written to a file.

I/O Link Compression
--------------------

I/O Link Compression enables dOpenCL's transfers between the host and the nodes
to be compressed on-the-fly, i.e. at the time when the transfer is done.
This can provide a significant speed-up for data transfers if the network
transfer speed becomes a limiting performance bottleneck.

I/O Link Compression is implemented using the 842 Compression Algorithm,as
implemented in lib842. lib842 provides optimized software, GPU and hardware
accelerator implementations for 842 Compression, which can efficiently provide
speeds sufficient to saturate moderate-to-high speed networks (e.g. 10Gbit/s).

More details about this algorithm and those implementations are available in
the lib842 repository at https://github.com/joanbm/lib842.
lib842 is provided as a submodule of the dOpenCL repository, so its sources
are automatically pulled in when a clone including submodules is done
(git clone --recurse-submodules).

I/O Link Compression is transparent from the application's point of view,
i.e. no re-compilation or code changes to the application code are necessary
to take advantage of it.
However, note that both the dOpenCL host and daemons must be compiled with I/O
Link Compression support, in order to be able to work correctly together.

The following compile-time flags are available regarding I/O Link Compression.
Those flags can be set when building using CMake:
* ENABLE_IO_LINK_COMPRESSION: Enable I/O Link Compression (using lib842) for
                              transfers
* USE_HW_IO_LINK_COMPRESSION: Use in-kernel and potentially hardware-accelerated
                              842 implementation (for both compression and
                              decompression, using the cryptodev kernel module)
* USE_CL_IO_LINK_COMPRESSION: Use OpenCL GPU-accelerated 842 implementation
                              (Since lib842 currently only provides a 842 GPU
                              decompressor, this only affects decompression)

Once dOpenCL has been built with I/O Link Compression, it will be used by
default using the specified configuration and reasonably sane defaults.
However, the behaviour can be changed at runtime for testing or performance
tuning using the following environment variables:
* DCL_DISABLE_IO_LINK_COMPRESSION=(any): Disables I/O Link Compression (after it
                                         has been built in). Both the host and
                                         all the daemons must specify this
                                         environment variable, if used.
* DCL_DISABLE_HW_IO_LINK_COMPRESSION=(any): Disables the in-kernel and
                                            potentially hardware-accelerated 842
                                            implementation (after if it has been
                                            built in, and even if available).
* DCL_DISABLE_CL_IO_LINK_COMPRESSION=(any): Disables the OpenCL GPU-accelerated
                                            842 implementation
                                            (after if it has been built in,
                                            and even if available).
* DCL_CL_IO_LINK_COMPRESSION_INLINE=(any): Uses the alternative OpenCL in-place
                                           842 implementation (unstable)
* DCL_IO_LINK_NUM_COMPRESS_THREADS=(number): Sets the number of threads used
                                             for compression.
                                             (Otherwise, a thread per hardware
                                             thread is used by default)
* DCL_IO_LINK_NUM_DECOMPRESS_THREADS=(number): Sets the number of threads used
                                               for decompression.
                                               (Otherwise, a thread per hardware
                                               thread is used by default)

-----------------
Project structure
-----------------

/
+- dcl		                dOpenCL API definitions
|  +- doc			Doxygen documentation of dOpenCL API
|  +- include
|  |  +- CL                     OpenCL API extension
|  |  |  cl_wwu_collective.h    collective operations
|  |  |  cl_wwu_dcl.h           distributed OpenCL
|  |  +- dcl                    dOpenCL API headers
|  |        ComputeNode.h
|  |        CommunicationManager.h
|  |        Device.h
|  |        Host.h
|  |        ...
+- dclasio                      dOpenCL API implementation using Boost.Asio
+- icdpp                        ICD implementation (C++, Boost.Asio only)
|                               implements OpenCL (including API extension) using dOpenCL API
+- daemon                       dOpenCL daemon
+- test                         Test suite (based on Boost Test Library, experimental)


------------
Known Issues
------------

* dOpenCL does not yet support the OpenCL ICD loader mechanism.
  Hence, dOpenCL has to be preloaded using LD_PRELOAD to override the system's
  ICD loader or OpenCL implementation.

* In most cases, the dOpenCL daemon will crash if a host is disconnected (e.g.,
  due to an application failure) during a running data transfer.

* Programs currently cannot be built from binaries or built-in kernels (OpenCL
  1.2)

* Programs are always build synchronously; callbacks are supported though

* #include directives in OpenCL C programs are currently not supported

* dOpenCL does not support the following OpenCL APIs, but will support them in
  future releases:
  + all image and sampler APIs
  + all sub-buffer APIs
  + vendor-specific extensions (e.g., device fission)
  + all sub-device APIs (OpenCL 1.2)
  + clGetKernelArgInfo (OpenCL 1.2)
  + cl{Compile|Link}Program (OpenCL 1.2)
  + clEnqueue{Read|Write|Copy}BufferRect (OpenCL 1.2)
  + clEnqueueFillBuffer (OpenCL 1.2)
  + clEnqueueMigrateMemObject (OpenCL 1.2)

* dOpenCL does not and will not support the following OpenCL APIs:
  + OpenGL/CL or DirectX/CL interop
  + native kernels (clEnqueueNativeKernel is implemented but always returns
    CL_INVALID_OPERATION)
  + OpenCL 1.0 deprecated APIs

* dOpenCL collective operation APIs are not yet available

* Communication between nodes in dOpenCL is not secured in any way.

* Nodes in dOpenCL should all use the same byte order.


-------
Contact
-------

If you have any questions or suggestions regarding dOpenCL feel free to contact
me via email (philipp.kegel@uni-muenster.de).

