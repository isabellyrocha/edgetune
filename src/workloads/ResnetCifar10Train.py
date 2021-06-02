from tuning import InferenceServer
from workloads.models.Resnet import Resnet
from keras.datasets import cifar10
from tensorflow import keras
from pathlib import Path
from utils import utils
import tensorflow as tf
from ray import tune
import json
import shutil
import time
import os

#tf.config.threading.set_inter_op_parallelism_threads(8)
#tf.config.threading.set_intra_op_parallelism_threads(8)

class ResnetCifar10Train(tune.Trainable):

    def setup(self, config):
        self.timestep = 0
        self.inference_duration = None
        self.inference_batch = None
        self.inference_cores = None

    def step(self):
        self.timestep += 1

        ##### Setting training configurations #####
        n = self.config.get("n", 3)
        print(n)
        model_depth = n * 6 + 2
        train_batch = self.config.get("train_batch", 128)
        #utils.set_cores(self.config.get("train_cores", 8))
        #utils.set_memory(self.config.get("train_memory", 64))

        ##### Dataset #####
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        shape = x_train.shape[1:]
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        ##### Model #####
        res = Resnet()
        model = res.resnet_v1(input_shape=shape, depth=model_depth)
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        directory_name = "%s/edgetune/models/model_%d" % (str(Path.home()), n)

        if os.path.isdir(directory_name):
            model = keras.models.load_model(directory_name)

        ##### Training #####
        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

        epochs = 1
        training_start = time.time()
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
                break
        
        training_duration = time.time() - training_start
        train_acc = train_acc_metric.result()
        training_accuracy = float(train_acc_metric.result())
        train_acc_metric.reset_states()

        if os.path.isdir(directory_name):
            shutil.rmtree(directory_name)
        os.mkdir(directory_name)
        model.save(directory_name)
        
        if self.inference_duration is None:
            accResults = InferenceServer.runSearch(n)
            self.inference_duration = accResults['inference_duration']
            self.inference_cores = accResults['config']['inference_cores']
            self.inference_batch = accResults['config']['inference_batch']

        runtime_ratio = (training_duration*self.inference_duration)/training_accuracy
        
        result = {
            "runtime_ratio": runtime_ratio,
            "training_accuracy": training_accuracy,
            "training_duration": training_duration,
            "inference_duration": self.inference_duration,
            "inference_cores": self.inference_cores,
            "inference_batch": self.inference_batch
        }

        return result

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({
                "timestep": self.timestep,
                "inference_duration": self.inference_duration,
                "inference_batch": self.inference_batch,
                "inference_cores": self.inference_cores
            }))
        return path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path) as f:
            j = json.loads(f.read())
            self.timestep = j["timestep"]
            self.inference_duration = j["inference_duration"]
            self.inference_batch = j["inference_batch"]
            self.inference_cores = j["inference_cores"]
