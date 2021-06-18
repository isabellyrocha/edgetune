"""
Run training, inference, or ONNX inference of torchvision models
"""

import time
import argparse
import lib.rapl.rapl as rapl

arguments = argparse.ArgumentParser()
arguments.add_argument(
    "--model",
    default="resnet18",
    choices=[
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152",
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
arguments.add_argument("--function", default="train", choices=["train", "train_epoch", "train_dataset", "train_multi", "inference", "onnx_inference"])
arguments.add_argument("--time", default=1, type=int)
arguments.add_argument("--batch", default=1024, type=int)
arguments.add_argument("--size", default=256, type=int)
arguments.add_argument("--gpus", default=1, type=int)
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

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# minimal training loop
# training continues until args.time amount of seconds of the forward pass has been gathered
def train(loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    pbar = tqdm(total=args.time)
    epochs = 200
    correct = 0
    total = 0
    start = time.time()
    start_energy = rapl.RAPLMonitor.sample()
    for epoch in range(epochs):
        for (images, target) in loader:
            images, target = images.to(device), target.to(device)
            
            # Forward Phase
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            loss = criterion(out, target)

            elapsed = time.time() - start
            pbar.update(elapsed)

            # Backward Phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = (100 * correct / total)
        print('[Epoch %d] %d seen samples with accuracy %d %%' % (epoch, total, accuracy))
        if accuracy >= 80:
            break
    end_energy = rapl.RAPLMonitor.sample()
    diff = end_energy-start_energy
    training_energy = diff.energy('package-0')
    print('Energy: %f' % training_energy)
    elapsed = time.time() - start
    print('Total elapsed time: %f' % elapsed)

def train_epoch(loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    pbar = tqdm(total=args.time)
    epochs = 200
    correct = 0
    total = 0
    start = time.time()
    start_energy = rapl.RAPLMonitor.sample()
    curr_epoch = 0
    trial = 1
    while curr_epoch < epochs:
        trial_epochs = trial * 2
        for epoch in range(trial_epochs):

            for (images, target) in loader:
                images, target = images.to(device), target.to(device)

                # Forward Phase
                out = model(images)
                _, predicted = torch.max(out.data, 1)
                loss = criterion(out, target)

                elapsed = time.time() - start
                pbar.update(elapsed)

                # Backward Phase
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += target.size(0)
                correct += (predicted == target).sum().item()
                accuracy = (100 * correct / total)
        curr_epoch += trial_epochs
        print('[Trial %d][Epoch %d] %d seen samples with accuracy %d %%' % (trial, curr_epoch, total, accuracy))
        trial += 1
        if accuracy >= 80:
            break
    end_energy = rapl.RAPLMonitor.sample()
    diff = end_energy-start_energy
    training_energy = diff.energy('package-0')
    print('Energy: %f' % training_energy)
    elapsed = time.time() - start
    print('Total elapsed time: %f' % elapsed)

def get_percentage(step):
    if step >= 10:
        return 1
    return step*0.1

def train_dataset(loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    pbar = tqdm(total=args.time)
    epochs = 1000
    correct = 0
    total = 0
    all_images = 50000
    start = time.time()
    start_energy = rapl.RAPLMonitor.sample()
    for epoch in range(epochs):
        percentage = get_percentage(epoch)
        epoch_images = all_images * percentage
        for (images, target) in loader:
            images, target = images.to(device), target.to(device)

            # Forward Phase
            out = model(images)
            _, predicted = torch.max(out.data, 1)
            loss = criterion(out, target)

            elapsed = time.time() - start
            pbar.update(elapsed)

            # Backward Phase
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = (100 * correct / total)

            if total >= epoch_images:
                break
        print('[Trial %d][Epoch %d] %d seen samples with accuracy %d %%' % (epoch, epoch, total, accuracy))
        if accuracy >= 80:
            break
    end_energy = rapl.RAPLMonitor.sample()
    diff = end_energy-start_energy
    training_energy = diff.energy('package-0')
    print('Energy: %f' % training_energy)
    elapsed = time.time() - start
    print('Total elapsed time: %f' % elapsed)

def train_multi(loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.train()
    pbar = tqdm(total=args.time)
    max_epochs = 1000
    correct = 0
    total = 0
    all_images = 50000
    start = time.time()
    start_energy = rapl.RAPLMonitor.sample()
    curr_epochs = 0
    trials = 1
    while curr_epochs < max_epochs:
        trial_epochs = trials * 2

        percentage = get_percentage(trials)
        epoch_images = all_images * percentage

        for epoch in range(trial_epochs):
            for (images, target) in loader:
                images, target = images.to(device), target.to(device)

                # Forward Phase
                out = model(images)
                _, predicted = torch.max(out.data, 1)
                loss = criterion(out, target)

                elapsed = time.time() - start
                pbar.update(elapsed)

                # Backward Phase
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total += target.size(0)
                correct += (predicted == target).sum().item()
                accuracy = (100 * correct / total)

                if total >= epoch_images:
                    break
        curr_epochs += trial_epochs
        print('[Trial %d][Epoch %d] %d seen samples with accuracy %d %%' % (trials, curr_epochs, total, accuracy))
        trials += 1
        if accuracy >= 80:
            break

    end_energy = rapl.RAPLMonitor.sample()
    diff = end_energy-start_energy
    training_energy = diff.energy('package-0')
    print('Energy: %f' % training_energy)
    elapsed = time.time() - start
    print('Total elapsed time: %f' % elapsed)


@torch.no_grad()  # disable gradients
def inference(loader, model):
    total_images= 0
    model.eval()
    start = time.time()
    start_energy = rapl.RAPLMonitor.sample()
    for (images, target) in loader:
        out = model(images)
        _, pred = torch.max(out.data, 1)

        total_images += len(images)
        if total_images >= 1000:
            elapsed = time.time() - start
            print("Elapsed time: %f" % elapsed)
            end_energy = rapl.RAPLMonitor.sample()
            diff = end_energy-start_energy
            training_energy = diff.energy('package-0')
            print('Energy: %f' % training_energy)
            break

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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not args.baseline_mem:
    model = models.__dict__[args.model]()
    model = torch.nn.DataParallel(model, device_ids = list(range(args.gpus)))
    model.to(device)

loader = torch.utils.data.DataLoader(    
    datasets.CIFAR10(
        ".cache/", transform=T.Compose([T.Resize(args.size), T.CenterCrop(int(args.size * 224 / 256)), T.ToTensor()])
    ),
    batch_size=args.batch,
)

print ("Starting %s with batch %d using %d GPUs..." % (args.function, args.batch, args.gpus))

if args.function == "train":
    fn = train
elif args.function == "train_epoch":
    fn = train_epoch
elif args.function == "train_dataset":
    fn = train_dataset
elif args.function == "train_multi":
    fn = train_multi
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
