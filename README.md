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

## Install cpupower (for frequency scaling)

```Shell
$ sudo apt-get install linux-cpupower
$ sudo apt-get install linux-tools-5.4.0-48-generic linux-cloud-tools-5.4.0-48-generic
```

# Install requirements for PyRAPL measurements

```Shell
$ python3 -m pip install pybluez
```
