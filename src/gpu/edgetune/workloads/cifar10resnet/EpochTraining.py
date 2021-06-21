from tuning import InferenceServer
from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import cifar10
from tensorflow import keras
from pathlib import Path
from utils import utils
import workloads.cifar10resnet.lib.rapl.rapl as rapl
import tensorflow as tf
from ray import tune
from threading import Thread
import json
import shutil
import time
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EpochTraining(tune.Trainable):
    def setup(self, config):
        self.steps = 0
        self.epochs = 0
        self.inference_duration = None
        self.inference_energy = None
        self.inference_batch = None
        self.inference_cores = None

    def step(self):
        self.steps += 1

        ##### Setting training configurations #####
        n = self.config.get("n", 3)
        model_depth = n * 6 + 2
        train_batch = self.config.get("train_batch", 128)
        utils.set_training_cores(self.config.get("train_cores", 4))

        ### Inference ###
        if self.inference_duration is None:
            inf_serv_results = {}
        #    InferenceServer.runSearch(n, inf_serv_results)
            self.inference_duration = 0#inf_serv_results['inference_duration']
            self.inference_energy = 0#inf_serv_results['inference_energy']
            self.inference_cores = 2#inf_serv_results['config']['inference_cores']
            self.inference_batch = 2#inf_serv_results['config']['inference_batch']

        ##### Dataset #####
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #x_train, y_train = x_train.to(device), y_train.to(device)
        #x_test, y_test = x_test.to(device), y_test.to(device)
        shape = x_train.shape[1:]
        x_train = x_train.astype('float32') / 255
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch)

        ##### Model #####
        res = Resnet()
        model = res.resnet_v1(input_shape=shape, depth=model_depth)
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        directory_name = "%s/edgetune/models/model_%d" % (str(Path.home()), n)

        if os.path.isdir(directory_name):
            model = keras.models.load_model(directory_name)
        model.to(device)

        ##### Training #####
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        epochs = 1
        self.epochs += epochs
        training_start = time.time()
        start_energy = rapl.RAPLMonitor.sample()
        for epoch in range(epochs):
            print("Start of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                step_start = time.time()
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)

                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y_batch_train, logits)

                if step % 200 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far: %s samples" % ((step + 1) * train_batch))
        
        training_duration = time.time() - training_start
        end_energy = rapl.RAPLMonitor.sample()
        diff = end_energy-start_energy
        training_energy = diff.energy('package-0')
        train_acc = train_acc_metric.result()
        training_accuracy = float(train_acc_metric.result())
        train_acc_metric.reset_states()

        model.save(directory_name)

        runtime_ratio = (training_duration*self.inference_duration)/training_accuracy
        
        result = {
            "epochs": self.epochs,
            "runtime_ratio": runtime_ratio,
            "training_accuracy": training_accuracy,
            "training_duration": training_duration,
            "training_energy": training_energy,
            "inference_duration": self.inference_duration,
            "inference_energy": self.inference_energy,
            "inference_cores": self.inference_cores,
            "inference_batch": self.inference_batch
        }

        return result

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({
                "steps": self.steps,
                "epochs": self.epochs,
                "inference_duration": self.inference_duration,
                "inference_energy": self.inference_energy,
                "inference_batch": self.inference_batch,
                "inference_cores": self.inference_cores
            }))
        return path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path) as f:
            j = json.loads(f.read())
            self.steps = j["steps"]
            self.epochs = j["epochs"]
            self.inference_duration = j["inference_duration"]
            self.inference_energy = j["inference_energy"]
            self.inference_batch = j["inference_batch"]
            self.inference_cores = j["inference_cores"]
