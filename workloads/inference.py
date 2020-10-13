"""
Run training, inference, or ONNX inference of torchvision models
"""

import argparse
import os
import time
import torch
import torchvision
from torchvision import datasets, transforms as T, models as models
import torch.onnx
import onnx
#import onnxruntime
from tqdm import tqdm
import pyRAPL # for energy measurements
import numpy as np

@torch.no_grad()  # disable gradients
def inference(loader, model):
    meter = pyRAPL.Measurement('metrics')
    meter.begin()
    model.eval()
    images,target = next(iter(loader))
    out = model(images)
    _, pred = torch.max(out.data, 1)
    print(pred)
    model.eval()
    meter.end()
    return meter.result

if __name__ == "__main__":
    classification_models = [
            "alexnet",
            "vgg11",
            "vgg11_bn",
            "vgg13",
            "vgg13_bn",
            "vgg16",
            "vgg16_bn",
            "vgg19",
            "vgg19_bn",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
            "squeezenet1_0",
            "squeezenet1_1",
            "densenet121",
            "densenet169",
            "densenet161",
            "densenet201",
            "inception_v3",
            "googlenet",
            "shufflenet_v2_x0_5",
            "shufflenet_v2_x1_0",
#            "shufflenet_v2_x1_5",
#            "shufflenet_v2_x2_0",
            "mobilenet_v2",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
            "mnasnet0_5",
#            "mnasnet0_75",
            "mnasnet1_0",
#            "mnasnet1_3"
    ]

    segmentation_models = [
            "fcn_resnet50",
            "fcn_resnet101",
            "deeplabv3_resnet50",
            "deeplabv3_resnet101",
    ]

    detection_models = [
            "fasterrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn",
            "maskrcnn_resnet50_fpn",
            "keypointrcnn_resnet50_fpn",
    ]

    video_models = [
            "r3d_18",
            "mc3_18",
            "r2plus1d_18"
    ]


    pyRAPL.setup()

    for model_name in classification_models:
        model = models.__dict__[model_name](pretrained=True)
    
        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=".cache/", train=False, download=True , transform=T.Compose([T.Resize(256), T.CenterCrop(int(256 * 224 / 256)), T.ToTensor()])
            ),
            batch_size=32
        )

        result = inference(loader, model)
        print("%s,cifar10,32,%f,%f,%f" % (model_name, result.duration/1000000, np.sum(result.pkg)/1000000, np.sum(result.dram)/1000000))

        loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=".cache/", train=False, download=True , transform=T.Compose([T.Resize(256), T.CenterCrop(int(256 * 224 / 256)), T.ToTensor()])
            ),
            batch_size=32
        )

        result = inference(loader, model)
        print("%s,cifar100,32,%f,%f,%f" % (model_name, result.duration/1000000, np.sum(result.pkg)/1000000, np.sum(result.dram)/1000000))

