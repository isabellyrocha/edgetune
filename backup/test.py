"""
Run training, inference, or ONNX inference of torchvision models
"""

import time

print("init")

import argparse

arguments = argparse.ArgumentParser()
arguments.add_argument(
    "--model",
    default="resnet18",
    choices=[
        "resnet18",
        "alexnet",
        "vgg16",
        "squeezenet1_0",
        "shufflenet_v2_x1_0",
        "mobilenet_v2",
        "resnext50_32x4d",
        "wide_resnet50_2",
        "mnasnet1_0",
    ],
)
arguments.add_argument("--function", default="train", choices=["train", "inference", "onnx_inference"])
arguments.add_argument("--time", default=1, type=int)
arguments.add_argument("--batch", default=1024, type=int)
arguments.add_argument("--size", default=256, type=int)
arguments.add_argument("--baseline_mem", default=False, action="store_true")
args = arguments.parse_args()

import os
import time
import torch
from torchvision import datasets, transforms as T, models as models
import torch.onnx
import onnx
import onnxruntime
from tqdm import tqdm

# minimal training loop
# training continues until args.time amount of seconds of the forward pass has been gathered
def train(loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    fwd_time = 0
    pbar = tqdm(total=args.time)
    epochs = 1
    correct = 0
    total = 0
    start = time.time()
    for epoch in range(epochs):
        for (images, target) in loader:
            #start = time.time()

            print("forward")  # workloads should print the phase they are in when starting them
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            loss = criterion(out, target)

            elapsed = time.time() - start
            fwd_time += elapsed
            pbar.update(elapsed)

            print("backward")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += target.size(0)
            correct += (predicted == target).sum().item()

            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    elapsed = time.time() - start
    print('Elapsed time: %d' & elapsed)


@torch.no_grad()  # disable gradients
def inference(loader, model):
    model.eval()
    fwd_time = 0
    pbar = tqdm(total=args.time)
    while fwd_time < args.time:
        for (images, target) in loader:
            start = time.time()

            print("forward")
            out = model(images)
            _, pred = torch.max(out.data, 1)

            elapsed = time.time() - start
            fwd_time += elapsed
            pbar.update(elapsed)
            if fwd_time > args.time:
                return


@torch.no_grad()
def onnx_inference(loader, ort_session):
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
                return


if not args.baseline_mem:
    model = models.__dict__[args.model]()

loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        ".cache/", transform=T.Compose([T.Resize(args.size), T.CenterCrop(int(args.size * 224 / 256)), T.ToTensor()])
    ),
    batch_size=args.batch,
)

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
    fn(loader, model)
