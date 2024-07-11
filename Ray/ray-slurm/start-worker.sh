#!/bin/bash

# HJ
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8

set -x

echo "starting ray worker node"
ray start --address $1
sleep infinity
