#!/bin/bash

# HJ
# export LC_ALL=C.UTF-8
# export LANG=C.UTF-8
set -x

echo "starting ray head node"
# Launch the head node
ray start --head
#ray start --head --node-ip-address=$1 --port=6379
sleep infinity
