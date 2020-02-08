#!/usr/bin/env bash

if status --is-login
    set -gx PATH $PATH "/opt/dopencl/bin"
    set -gx LIBRARY_PATH $LIBRARY_PATH "/opt/dopencl/lib"
    set -gx LD_LIBRARY_PATH $LD_LIBRARY_PATH "/opt/dopencl/lib"
    set -gx C_INCLUDE_PATH $C_INCLUDE_PATH "/opt/dopencl/include"
    set -gx CPLUS_INCLUDE_PATH $CPLUS_INCLUDE_PATH "/opt/dopencl/include"
end
