#!/bin/bash

echo "starting ray worker node"
ray start --address $1 --redis-password=$2
sleep infinity