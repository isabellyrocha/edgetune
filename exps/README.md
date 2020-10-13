# Experiments

## Architectures

Machine | chasseral | chaumont | vully | nuc | scopi
------- | --------- | -------- | ----- | --- | -----
Model   | ARMv7 Processor rev 4 (v7l) | ARMv7 Processor rev 4 (v7l) | Raspberry Pi 3 Model B Plus Rev 1.3 | Intel(R) Core(TM) i7-7567U CPU | ARMv8 Processor rev 1 (v8l)
Cores   | 4 | 4 | 4 | 4 | 4
CPU Frequency | 600 MHz - 1.50 GHz | 600 MHz - 1.50 GHz | 700 MHz - 1.40 GHz | 400 MHz - 4.00 GHz | -
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
  - shufflenet_v2_x1_5
  - shufflenet_v2_x2_0
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
  - mnasnet0_75: depth multiplier of 0.75.
  - mnasnet1_0: depth multiplier of 1.0. 
  - mnasnet1_3: MNASNet with depth multiplier of 1.3.

## Inferece (Min Frequency, batch size = 64)

- nuc < Chasseral ~ Chaumont < vully 

network            | chasseral          | chaumont           | vully              | nuc                
------------------ | ------------------ | ------------------ | ------------------ | ------------------
resnet18           | 24.8  | 26.4  | 40.2  | 15.2
alexnet            | 8.1   | 8.1  | 13.0 | 5.3
vgg16              | 101.7  | 104.0 | -                  | 108.4
squeezenet1_0      | 27.5 | 29.6 | 46.6 | 13.1
shufflenet_v2_x1_0 | 19.6 | 20.9  | 30.2  | 3.6
mobilenet_v2       | 45.6   | 46.1  | 71.7  | 14.0
resnext50_32x4d    | 199.6 | 207.4  | -                  | 45.6
wide_resnet50_2    | 240.7 | 240.7 | -                  | 94.9
mnasnet1_0         | 45.9  |  45.9 | 66.8  | 10.3

## Inferece (Max Frequency, batch size = 64)

- Scaling CPU frequency reduces time by ~half.
- Longer running inferences benefit more from scaling when using the nuc.

network            | chasseral          | chaumont           | vully              | nuc
------------------ | ------------------ | ------------------ | ------------------ | ------------------
resnet18           | 11.5 | 11.5 | 23.0 | 1.9
alexnet            | 3.4  | 3.7  | 7.6 | 1.2
vgg16              | 47.5  | 50.3  | -                  | 13.2
squeezenet1_0      | 13.7  | 14.4 | 29.1 | 1.8
shufflenet_v2_x1_0 | 9.6  | 10.0 | 18.1  | 1.4
mobilenet_v2       | 21.2 | 22.3 | 45.9  | 1.8
resnext50_32x4d    | 91.8  | 96.6  | -                  | 7.4
wide_resnet50_2    | 112.8  | 116.5 | -                  | 12.1
mnasnet1_0         | 21.0 | 22.4  | 38.2  | 7.5

## Inferece (Max Frequency, batch size = 32)

- Reducing batch size reduces time.
- Fixed the memory problem for resnext50_32x4d and wide_resnet50_2, but not for vgg16.

network            | chasseral          | chaumont           | vully              | nuc
------------------ | ------------------ | ------------------ | ------------------ | ------------------
resnet18           | 6.2  | 6.4  | 12.5 | 1.6
alexnet            | 2.0  | 2.1 | 4.0  | 0.6
vgg16              | 25.4 | 26.5 | -                  | 6.6
squeezenet1_0      | 7.3  | 7.9  | 1.5 | 0.9
shufflenet_v2_x1_0 | 4.6 | 5.5 | 9.9  | 0.8
mobilenet_v2       | 11.4 | 11.3 | 11.3 | 1.6
resnext50_32x4d    | 50.3 | 52.2  | 140.8    | 3.4
wide_resnet50_2    | 60.5   | 62.3  | 119.7 | 5.5
mnasnet1_0         | 49.9 | 51.8 | 119.2 | 3.2
