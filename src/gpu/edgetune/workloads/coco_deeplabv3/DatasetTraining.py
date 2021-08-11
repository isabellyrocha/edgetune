from tuning import InferenceServer
from workloads.cifar10resnet.Resnet import Resnet
from keras.datasets import cifar10
import torchvision.datasets as dset
import torchvision.transforms as transforms
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
from pycocotools.coco import COCO

class DatasetTraining(tune.Trainable):
    def setup(self, config):
        self.epochs = 0
        self.inference_duration = None
        self.inference_energy = None
        self.inference_batch = None
        self.inference_cores = None

    def get_percentage(self, step):
        if step >= 10:
            return 1
        return step*0.1

    def step(self):
        ##### Setting training configurations #####
        chosen_model = self.config.get("model", "deeplabv3_resnet50")
        train_batch = self.config.get("train_batch", 128)
        utils.set_training_cores(self.config.get("train_cores", 4))

        #### Inference ###
        inf_serv_results = {}
        inf_serv_results = InferenceServer.runSearch(chosen_model, inf_serv_results)
        self.inference_duration = inf_serv_results['inference_duration']
        self.inference_energy = inf_serv_results['inference_energy']
        self.inference_cores = inf_serv_results['config']['inference_cores']
        self.inference_batch = inf_serv_results['config']['inference_batch']

        ##### Dataset #####
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
        #shape = x_train.shape[1:]
        #x_train = x_train.astype('float32') / 255
        #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch)
        #train_dataset = dset.CocoCaptions(root = '%s/coco/train2017' % Path.home(),
        #                        annFile ='%s/coco/annotations/captions_train2017.json' % Path.home(),
        #                        transform=transforms.ToTensor())
        coco = CocoDataset(root = '%s/coco/train2017' % Path.home(),
                           json ='%s/coco/annotations/captions_train2017.json' % Path.home(),
                           transform=transforms.ToTensor())

        loader = torch.utils.data.DataLoader(
            dataset=coco,
            batch_size=args.batch,
        )


        ##### Model #####
        #res = Resnet()
        #model = res.resnet_v1(input_shape=shape, depth=model_depth)
        #optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        #loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        #directory_name = "%s/edgetune/models/model_%d" % (str(Path.home()), n)

        #if os.path.isdir(directory_name):
        #    model = keras.models.load_model(directory_name)

        model = models.__dict__[chosen_model]()
        model = torch.nn.DataParallel(model, device_ids = list(range(self.config.get("gpus", 8))))
        model.to(device)

        ##### Training #####
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        epochs = 1
        self.epochs += epochs
        percentage = self.get_percentage(self.epochs)
        total_images = 5000 * percentage
        training_start = time.time()
        start_energy = rapl.RAPLMonitor.sample()
        for epoch in range(epochs):
            print("Start of epoch %d" % (epoch,))

            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                if step*train_batch >= total_images:
                    break

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
###########
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        pbar = tqdm(total=args.time)
        epochs = 1
        self.epochs += epochs
        percentage = self.get_percentage(self.epochs)
        correct = 0
        total = 0
        start = time.time()
        #start_energy = rapl.RAPLMonitor.sample()
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
        #end_energy = rapl.RAPLMonitor.sample()
        #diff = end_energy-start_energy
        #training_energy = diff.energy('package-0')
        #print('Energy: %f' % training_energy)
        elapsed = time.time() - start
        print('Total elapsed time: %f' % elapsed)

        
        result = {
            "epochs": self.epochs,
            #"runtime_ratio": runtime_ratio,
            #"training_accuracy": training_accuracy,
            #"training_duration": training_duration,
            #"training_energy": training_energy,
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
            self.epochs = j["epochs"]
            self.inference_duration = j["inference_duration"]
            self.inference_energy = j["inference_energy"]
            self.inference_batch = j["inference_batch"]
            self.inference_cores = j["inference_cores"]
