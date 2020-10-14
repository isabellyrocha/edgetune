# EdgeTune

## Requirements:

- pip3
- torch
- torchvision

## Install pip3
```Shell
$ sudo apt-get update
$ sudo apt-get install python3-pip
```

## Install torch and torchvision (Intel)

```Shell
$ pip3 install torch torchvision 
```

## Install torch and torchvision (ARM)
```Shell
$ git clone https://github.com/Ben-Faessler/Python3-Wheels.git
$ sudo pip3 install Python3-Wheels/pytorch/torch-1.5.0a0+4ff3872-cp37-cp37m-linux_armv7l.whl
$ sudo pip3 install Python3-Wheels/torchvision/torchvision-0.6.0a0+b68adcf-cp37-cp37m-linux_armv7l.whl
```

## Install torch and torchvision (Jetson)
```Shell
$ wget https://nvidia.box.com/shared/static/mmu3xb3sp4o8qg9tji90kkxl1eijjfc6.whl -O torch-1.1.0-cp36-cp36m-linux_aarch64.whl
$ pip3 install numpy torch-1.1.0-cp36-cp36m-linux_aarch64.whl
$ sudo apt-get install libjpeg-dev zlib1g-dev
$ git clone -b v0.3.0 https://github.com/pytorch/vision torchvision
$ cd torchvision
$ sudo python setup.py install
```

## Install cpupower (for frequency scaling)

```Shell
$ sudo apt-get install linux-cpupower
$ cpufreq-info              # get cpu frequency information
$ cpufreq-set -u min -d max # set cpu frequency
```

## Install requirements for PyRAPL measurements

```Shell
$ pip3 install pymongo pandas
$ pip3 install pyRAPL
```

## Power Measurements Setup

# InfluxDB

```Shell
$ sudo apt install influxdb
$ service influxdb start
$ influx
$ CREATE DATABASE power
```
