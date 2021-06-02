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
        
        ##### Inference #####
        '''
        val_batch_size = self.config.get("train_inference", 32)
        val_dataset = val_dataset.batch(val_batch_size)
        utils.set_cores(self.config.get("inference_cores", 8))
        utils.set_memory(self.config.get("inference_memory", 16))

        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        inference_start = time.time()
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            #print(val_logits)
            val_acc_metric.update_state(y_batch_val, val_logits)

        inference_accuracy = float(val_acc_metric.result())
        val_acc_metric.reset_states()

        inference_duration = time.time() - inference_start
        real_ratio = inference_accuracy/inference_duration

        print("Inference accuracy: %f" % inference_accuracy)
        print("Inference duration: %f" % inference_duration)
        print("Real ratio: %f" % real_ratio)
        '''
        result = {
            #"runtime_ratio": 1
            "training_accuracy": training_accuracy,
            #"inference_accuracy": inference_accuracy,
            #"forward_duration": forward_duration,
            "training_duration": training_duration
            #"epoch_duration": epoch_duration,
            #"inference_duration": inference_duration,
            #"proxy_ratio": proxy_ratio,
            #"real_ratio": real_ratio
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
