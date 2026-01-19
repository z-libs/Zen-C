#!/bin/bash

## Downloads lua-5.4.2 headers (.h) and lib (.a) into ./lua

wget "https://sourceforge.net/projects/luabinaries/files/5.4.2/Linux%20Libraries/lua-5.4.2_Linux54_64_lib.tar.gz"

[[ -d "lua" ]] && rm -rf lua
mkdir -p lua

tar -xf lua-5.4.2_Linux54_64_lib.tar.gz

mv include/* *.a lua/
rm -rf *.so include lua-5.4.2_Linux54_64_lib.tar.gz
