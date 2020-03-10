#!/usr/bin/env bash
set -euo pipefail

# Build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/dopencl ..
make -j$(nproc)

# Remove old install
sudo rm -Rf /opt/dopencl /etc/profile.d/dopenclenv.sh /usr/share/fish/vendor_conf.d/dopenclenv.fish

# Add new install
sudo make install
sudo install -D -m755 ../dopenclenv.sh /etc/profile.d/dopenclenv.sh
sudo install -D -m755 ../dopenclenv.fish /usr/share/fish/vendor_conf.d/dopenclenv.fish
