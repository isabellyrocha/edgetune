#!/bin/bash

python3 hyperband_v2.py &
taskset -pc 0-7 $!
