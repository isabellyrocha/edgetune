#from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import cifar10
from tensorflow import keras
from pathlib import Path
from utils import utils
from torchvision import datasets, transforms as T, models as models
import lib.rapl.rapl as rapl
import tensorflow as tf
from ray import tune
from threading import Thread
import json
import shutil
import time
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiFidelityTraining(tune.Trainable):
    def setup(self, config):
        self.steps = 0
        self.epochs = 0
#        self.inference_duration = None
#        self.inference_energy = None
#        self.inference_batch = None
#        self.inference_cores = None
    
    def get_percentage(self, step):
        if step >= 10:
            return 1
        return step*0.1

    def step(self):
        self.steps += 1

        ##### Setting training configurations #####
        #n = self.config.get("n", 3)
        #model_depth = n * 6 + 2
        train_batch = self.config.get("training_batch", 128)
        utils.set_training_cores(self.config.get("train_cores", 4))

        ### Inference ###
#        if self.inference_duration is None:
#            inf_serv_results = {}
#            InferenceServer.runSearch(self.config.get("model", "resnet18"), inf_serv_results)
#            self.inference_duration = inf_serv_results['inference_duration']
#            self.inference_energy = inf_serv_results['inference_energy']
#            self.inference_cores = inf_serv_results['config']['inference_cores']
#            self.inference_batch = inf_serv_results['config']['inference_batch']

        ##### Dataset #####
#        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
#        shape = x_train.shape[1:]
#        x_train = x_train.astype('float32') / 255
#        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch)
        size = 2
        loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                ".cache/", transform=T.Compose([T.Resize(size), T.CenterCrop(int(size * 224 / 256)), T.ToTensor()]),
                download=True,
            ),
            batch_size=self.config.get("training_batch", 128),
        )
        

        ##### Model #####
        directory_name = "%s/edgetune/models/%s" % (str(Path.home()), self.config.get("model", "resnet18"))
        if os.path.isdir(directory_name):
            #model = keras.models.load_model(directory_name)
            model = torch.load(directory_name)
        else:
            model = models.__dict__[self.config.get("model", "resnet18")]()

        model = torch.nn.DataParallel(model, device_ids = list(range(self.config.get("gpus", 8))))
        model.to(device)       
        model.train()

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        ##### Training #####
        epochs = self.steps * 2
        self.epochs += epochs
        percentage = self.get_percentage(self.epochs)
        correct = 0
        total_images = 0
        max_images = 5000 * percentage

        training_start = time.time()
        start_energy = rapl.RAPLMonitor.sample() 
        for epoch in range(epochs):
            epoch_images = 0
            for (images, target) in loader:
                images, target = images.to(device), target.to(device)

                # Forward Phase
                out = model(images)
                _, predicted = torch.max(out.data, 1)
                loss = criterion(out, target)

                #elapsed = time.time() - straining_start
                #pbar.update(elapsed)

                # Backward Phase
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_images += target.size(0)
                correct += (predicted == target).sum().item()

            total_images += epoch_images
            training_accuracy = (100 * correct / total_images)
            if total_images >= max_images:
                break

        training_duration = time.time() - training_start
        end_energy = rapl.RAPLMonitor.sample()
        diff = end_energy-start_energy
        training_energy = diff.energy('package-0')

        #model.save(directory_name)
        torch.save(model.state_dict(), directory_name)

        #runtime_ratio = (training_duration*self.inference_duration)/training_accuracy
        
        result = {
            "epochs": self.epochs,
#            "runtime_ratio": runtime_ratio,
            "training_accuracy": training_accuracy,
            "training_duration": training_duration,
            "training_energy": training_energy
#            "inference_duration": self.inference_duration,
#            "inference_energy": self.inference_energy,
#            "inference_cores": self.inference_cores,
#            "inference_batch": self.inference_batch
        }

        return result

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({
                "steps": self.steps,
                "epochs": self.epochs,
#                "inference_duration": self.inference_duration,
#                "inference_energy": self.inference_energy,
#                "inference_batch": self.inference_batch,
#                "inference_cores": self.inference_cores
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
