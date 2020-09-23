#!/usr/bin/env bash
set -euo pipefail

echo -e "\033[0;36m**Building dOpenCL...**\033[0m"
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/dopencl ..
make -j"$(nproc)"

echo -e "\033[0;36m**Installing dOpenCL...**\033[0m"
# Remove old install
sudo rm -Rf /opt/dopencl /etc/profile.d/dopenclenv.sh /usr/share/fish/vendor_conf.d/dopenclenv.fish

# Add new install
sudo make install
sudo install -D -m755 ../dopenclenv.sh /etc/profile.d/dopenclenv.sh
sudo install -D -m755 ../dopenclenv.fish /usr/share/fish/vendor_conf.d/dopenclenv.fish
