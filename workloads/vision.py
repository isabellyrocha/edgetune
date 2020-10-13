"""
Run training, inference, or ONNX inference of torchvision models
"""

import argparse
import os
import time
import torch
from torchvision import datasets, transforms as T, models as models
import torch.onnx
import onnx
#import onnxruntime
from tqdm import tqdm
import pyRAPL # for energy measurements
import numpy as np

# minimal training loop
# training continues until args.time amount of seconds of the forward pass has been gathered
def train(loader, model):
    meter = pyRAPL.Measurement('metrics')
    meter.begin()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    fwd_time = 0
    pbar = tqdm(total=args.time)
    while fwd_time < args.time:
        for (images, target) in loader:
            start = time.time()

            print("forward")  # workloads should print the phase they are in when starting them
            out = model(images)
            loss = criterion(out, target)

            elapsed = time.time() - start
            fwd_time += elapsed
            pbar.update(elapsed)

            print("backward")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if fwd_time > args.time:
                meter.end()
                return meter.result

@torch.no_grad()  # disable gradients
def inference(loader, model):
    meter = pyRAPL.Measurement('metrics')
    meter.begin()
    model.eval()
    fwd_time = 0
    pbar = tqdm(total=args.time)
    while fwd_time < args.time:
        for (images, target) in loader:
            start = time.time()

            #print("forward")
            out = model(images)
            _, pred = torch.max(out.data, 1)

            elapsed = time.time() - start
            fwd_time += elapsed
            pbar.update(elapsed)
            if fwd_time > args.time:
                meter.end()
                return meter.result

@torch.no_grad()
def onnx_inference(loader, ort_session):
    meter = pyRAPL.Measurement('metrics')
    meter.begin()
    fwd_time = 0
    pbar = tqdm(total=args.time)
    while fwd_time < args.time:
        for (images, target) in loader:
            start = time.time()

            print("forward")
            ort_inputs = {ort_session.get_inputs()[0].name: images.numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            _, pred = torch.max(torch.tensor(ort_outs[0]), 1)

            elapsed = time.time() - start
            fwd_time += elapsed
            pbar.update(elapsed)
            if fwd_time > args.time:
                meter.end()
                return meter.result

if __name__ == "__main__":
    arguments = argparse.ArgumentParser()
    arguments.add_argument(
        "--model",
        default="resnet18",
        choices=[
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
            "shufflenet_v2_x1_5",
            "shufflenet_v2_x2_0",
            "mobilenet_v2",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
            "mnasnet0_5",
            "mnasnet0_75",
            "mnasnet1_0",
            "mnasnet1_3"
        ],
    )
    arguments.add_argument("--function", default="train", choices=["train", "inference", "onnx_inference"])
    arguments.add_argument("--time", default=1, type=int)
    arguments.add_argument("--batch", default=32, type=int)
    arguments.add_argument("--size", default=256, type=int)
    arguments.add_argument("--baseline_mem", default=False, action="store_true")
    args = arguments.parse_args()

    if not args.baseline_mem:
        #model = models.__dict__[args.model](pretrained=True)
        model = models.segmentation.__dict__["fcn_resnet50"](pretrained=True)
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            ".cache/", download=True, transform=T.Compose([T.Resize(args.size), T.CenterCrop(int(args.size * 224 / 256)), T.ToTensor()])
        ),
        batch_size=args.batch,
    )

    pyRAPL.setup()

    if args.function == "train":
        fn = train
    elif args.function == "inference":
        fn = inference
    elif args.function == "onnx_inference":
        fn = onnx_inference
        onnx_model_path = f"results/onnx/{args.model}_batch{args.batch}_size{args.size}.onnx"
        if not os.path.exists(onnx_model_path):
            # see https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html for more info on ONNX
            torch.onnx.export(
                model,
                torch.randn(
                    size=(args.batch, 3, int(args.size * 224 / 256), int(args.size * 224 / 256)), requires_grad=False
                ),
                onnx_model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            onnx_model = onnx.load(onnx_model_path)
            onnx.checker.check_model(onnx_model)
            model = onnxruntime.InferenceSession(onnx_model_path)

    if not args.baseline_mem:
        result = fn(loader, model)
        print("%s,%d,%f,%f,%f" % (args.model, args.batch, result.duration/1000000, np.sum(result.pkg)/1000000, np.sum(result.dram)/1000000))
