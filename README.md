# EdgeTune: Inferece-Oriented Parameter Tuning

## Requirements:

- pip3
- torch
- torchvision

## Install pip3
```Shell
$ sudo apt-get update
$ sudo apt-get install python3-pip
```

## Install torch and torchvision 

- Intel (nuc)
```Shell
$ pip3 install torch torchvision 
```

- ARM (RaspberryPi)
```Shell
$ git clone https://github.com/Ben-Faessler/Python3-Wheels.git
$ sudo pip3 install Python3-Wheels/pytorch/torch-1.5.0a0+4ff3872-cp37-cp37m-linux_armv7l.whl
$ sudo pip3 install Python3-Wheels/torchvision/torchvision-0.6.0a0+b68adcf-cp37-cp37m-linux_armv7l.whl
```

- Jetson Nano
```Shell
$ wget https://nvidia.box.com/shared/static/wa34qwrwtk9njtyarwt5nvo6imenfy26.whl -O torch-1.7.0-cp36-cp36m-linux_aarch64.whl
$ sudo apt-get install python3-pip libopenblas-base libopenmpi-dev 
$ pip3 install Cython
$ pip3 install numpy torch-1.7.0-cp36-cp36m-linux_aarch64.whl
$ sudo apt-get install libjpeg-dev zlib1g-dev
$ git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision 
$ cd torchvision
$ export BUILD_VERSION=0.8.1 
$ sudo python3 setup.py install
```

## Install cpupower (for frequency scaling)
```Shell
$ sudo apt-get install linux-cpupower
$ cpufreq-info              # get cpu frequency information
$ cpufreq-set -u min -d max # set cpu frequency
```

## Power Measurements Setup

- InfluxDB

```Shell
$ sudo apt install influxdb
$ service influxdb start
$ influx
$ CREATE DATABASE power
```

- PowerSpy Power
```Shell
$ pip install influxdb
$ python powerspy.py MAC
```

- POE Power
```Shell
$ apt-get install python-bluez bluez
$ hcitool scan 
$ hcitool dev 
$ systemctl restart bluetooth.service
$ pip3 install influxdb
$ python3 poe_power.py -p PORT
```

- RAPL

```Shell
$ sudo chmod -R a+r /sys/class/powercap/intel-rapl
```
