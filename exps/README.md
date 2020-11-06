# Experiments

## Architectures

Machine | chasseral | chaumont | vully | nuc | scopi
------- | --------- | -------- | ----- | --- | -----
Model   | ARMv7 Processor rev 4 (v7l) | ARMv7 Processor rev 4 (v7l) | Raspberry Pi 3 Model B Plus Rev 1.3 | Intel(R) Core(TM) i7-7567U CPU | ARMv8 Processor rev 1 (v8l)
Cores   | 4 | 4 | 4 | 4 | 4
CPU Frequency | 600 MHz - 1.50 GHz | 600 MHz - 1.50 GHz | 700 MHz - 1.40 GHz | 400 MHz - 4.00 GHz | 102 MHz - 1.48 GHZ
Memory  | 4GB | 4GB | 1GB | 16GB | 4GB
SSD     | 256 GB | - | - | - | -

## Models

Classification
- AlexNet (1)
  - alexnet
- VGG (8)
  - vgg11: 11-layer model 
  - vgg11_bn: 11-layer model with batch normalization
  - vgg13: 13-layer model
  - vgg13_bn: 13-layer model with batch normalization
  - vgg16: 16-layer model 
  - vgg16_bn: 16-layer model with batch normalization
  - vgg19: 19-layer model 
  - vgg19_bn: 19-layer model with batch normalization
- ResNet (5)
  - resnet18
  - resnet34
  - resnet50
  - resnet101
  - resnet152
- SqueezeNet(2)
  - queezenet1_0: SqueezeNet model architecture from the “SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size” paper.
  - squeezenet1_1: SqueezeNet 1.1 model from the official SqueezeNet repo. SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters than SqueezeNet 1.0, without sacrificing accuracy.
- DenseNet (4)
  - densenet121
  - densenet169
  - densenet161
  - densenet201
- Inception v3 (1)
  - inception_v3
- GoogLeNet (1)
  - googlenet: GoogLeNet (Inception v1) model architecture from “Going Deeper with Convolutions”.
- ShuffleNet v2 (4)
  - shufflenet_v2_x0_5
  - shufflenet_v2_x1_0
- MobileNet v2 (1)
  - mobilenet_v2: Constructs a MobileNetV2 architecture from “MobileNetV2: Inverted Residuals and Linear Bottlenecks”.
- ResNeXt (2)
  - resnext50_32x4d
  - resnext101_32x8d
- Wide ResNet (2)
  - wide_resnet50_2
  - wide_resnet101_2
- MNASNet (4)
  - mnasnet0_5: depth multiplier of 0.5. 
  - mnasnet1_0: depth multiplier of 1.0. 

## Models

- CIFAR100
- CIFAR10

## Performance Measurements

![alt text](https://github.com/isabellyrocha/edgetune/blob/main/exps/plots/duration.png?raw=true)

## Energy Measurements

![alt text](https://github.com/isabellyrocha/edgetune/blob/main/exps/plots/energy.png?raw=true)
