import workloads.cifar10resnet.lib.rapl.rapl as rapl
from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import cifar10
from tensorflow import keras
from pathlib import Path
from utils import utils
import tensorflow as tf
from ray import tune
import json
import time
import os

class Inference(tune.Trainable):
    def setup(self, config):
        self.timestep = 0

    def step(self):
        self.timestep += 1

        ##### Setting training configurations #####
        coco = CocoDataset(root = '%s/coco/train2017' % Path.home(),
                           json ='%s/coco/annotations/captions_train2017.json' % Path.home(),
                           transform=transforms.ToTensor())

        loader = torch.utils.data.DataLoader(
            dataset=coco,
            batch_size=args.batch,
        )


        model = models.__dict__[self.config.get("model", "deeplabv3_resnet50")]()
        model = torch.nn.DataParallel(model, device_ids = list(range(args.gpus)))



        ##### Inference #####
        val_batch_size = self.config.get("train_inference", 32)
        val_dataset = val_dataset.batch(val_batch_size)
        utils.set_inference_cores(self.config.get("inference_cores", 4))
 
        '''
        inference_start = time.time()
        start_energy = rapl.RAPLMonitor.sample()
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
        inference_duration = time.time() - inference_start
        end_energy = rapl.RAPLMonitor.sample()
        diff = end_energy-start_energy
        inference_energy = diff.energy('package-0')
        '''

        total_images= 0
        model.eval()
        start = time.time()
        #start_energy = rapl.RAPLMonitor.sample()
        for (images, target) in loader:
            out = model(images)
            _, pred = torch.max(out.data, 1)

            total_images += len(images)
            if total_images >= val_batch_size:
                inference_duration = time.time() - start
                print("Elapsed time: %f" % inference_duration)
                end_energy = rapl.RAPLMonitor.sample()
                diff = end_energy-start_energy
                inference_energy = diff.energy('package-0')
                print('Energy: %f' % training_energy)
                break

        result = {
            "inference_duration": inference_duration,
            "inference_energy": inference_energy
        }

        return result

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]
