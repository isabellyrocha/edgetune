import workloads.cifar10resnet.lib.rapl.rapl as rapl
from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import imdb
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
        n = self.config.get("n", 3)
        model_depth = n * 6 + 2

        ##### Dataset #####
        (x_train, y_train), (x_test, y_test) = imdb.load_data()
        shape = x_train.shape[1:]
        #x_test = x_test.astype('float32') / 255
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        ##### Model #####
        res = Resnet()
        model = res.resnet_v1(input_shape=shape, depth=model_depth)
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        ##### Inference #####
        val_batch_size = self.config.get("inference_batch", 1)
        val_dataset = val_dataset.batch(val_batch_size)
        utils.set_inference_cores(self.config.get("inference_cores", 4))

        inference_start = time.time()
        start_energy = rapl.RAPLMonitor.sample()
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
        inference_duration = time.time() - inference_start
        end_energy = rapl.RAPLMonitor.sample()
        diff = end_energy-start_energy
        inference_energy = diff.energy('package-0')

        result = {
            "n": n,
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
