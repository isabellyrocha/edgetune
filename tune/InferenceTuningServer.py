import subprocess
import resource
import argparse
import json
import os
import time
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
from keras.layers import AveragePooling2D, Input, Flatten
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
from keras.datasets import cifar10
import socket

class MyTrainableClass(tune.Trainable):

    def setup(self, config):
        self.timestep = 0
    
    def set_cores(self, cores):
        command = "ps -aux | awk '{print $2}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
        subprocess.Popen(['ssh', socket.gethostbyname(socket.gethostname()), command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Changed number of cores to %d... " % cores)

    def set_memory(self, memory):
        #command = "ulimit -Sv %d000000" % memory
        #subprocess.Popen(["ulimit -Sv %d000000" % memory], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #resource.setrlimit(resource.RLIMIT_VMEM, 1000000)
        subprocess.Popen('ulimit -Sv %d000000' % memory, shell=True)
        print("Changed memory to %dG... " % memory)

    def resnet_layer(self, inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
        conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
        layer = inputs
        
        if conv_first:
            layer = conv(layer)
            if batch_normalization:
                layer = BatchNormalization()(layer)
            if activation is not None:
                layer = Activation(activation)(layer)
        else:
            if batch_normalization:
                layer = BatchNormalization()(layer)
            if activation is not None:
                layer = Activation(activation)(layer)
            layer = conv(layer)

        return layer

    def resnet_v1(self, input_shape, depth, num_classes=10):
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
        
        # Start model definition.
        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self.resnet_layer(inputs=inputs)
        # Instantiate the stack of residual units
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    strides = 2  # downsample
                y = self.resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 strides=strides)
                y = self.resnet_layer(inputs=y,
                                 num_filters=num_filters,
                                 activation=None)
                if stack > 0 and res_block == 0:  # first layer but not first stack
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = self.resnet_layer(inputs=x,
                                     num_filters=num_filters,
                                     kernel_size=1,
                                     strides=strides,
                                     activation=None,
                                     batch_normalization=False)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        # Add classifier on top.
        # v1 does not use BN after last shortcut connection-ReLU
        x = AveragePooling2D(pool_size=8)(x)
        y = Flatten()(x)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        # Instantiate model.
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def step(self):
        self.timestep += 1

        ##### Setting training configurations #####
        n = self.config.get("n", 3)
        model_depth = n * 6 + 2

        ##### Dataset #####
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        shape = x_train.shape[1:]
        #x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        #train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        #train_dataset = train_dataset.shuffle(buffer_size=1024).batch(train_batch)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

        ##### Model #####
        model = self.resnet_v1(input_shape=shape, depth=model_depth)
        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        ##### Inference #####
        total_samples = 500
        val_batch_size = self.config.get("inference_batch", 32)
        val_dataset = val_dataset.batch(val_batch_size)
        self.set_cores(self.config.get("inference_cores", 4))
        self.set_memory(self.config.get("inference_memory", 16))

        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        inference_start = time.time()
        seen_samples = 0
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)
            seen_samples += val_batch_size
            if seen_samples > total_samples:
                break
        
        inference_accuracy = float(val_acc_metric.result())
        val_acc_metric.reset_states()
        
        inference_duration = time.time() - inference_start

        #print("Inference accuracy: %f" % inference_accuracy)
        #print("Inference duration: %f" % inference_duration)
        
        print("%d,%d,%d,%d,%f\n" % 
                (self.config.get("inference_cores", 8),
                self.config.get("inference_memory", 16),
                n,
                val_batch_size,
                inference_duration))

        return {"inference_duration": inference_duration}

    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "checkpoint")
        with open(path, "w") as f:
            f.write(json.dumps({"timestep": self.timestep}))
        return path

    def load_checkpoint(self, checkpoint_path):
        with open(checkpoint_path) as f:
            self.timestep = json.loads(f.read())["timestep"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    args, _ = parser.parse_known_args()
    
    tf.config.threading.set_inter_op_parallelism_threads(8)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    ray.init(num_cpus=8)

    tuning_start = time.time()

    hyperband = HyperBandScheduler(time_attr="inference_duration", metric="inference_duration", mode = "min", max_t=1)

    analysis = tune.run(
        MyTrainableClass,
        name="hyperband_test",
        num_samples=1,
        stop={"training_iteration": 1},
        resources_per_trial={
            "cpu": 8,
            "gpu": 0
        },
        config={
            "n": tune.grid_search([3, 5, 7]),
            "inference_cores": tune.grid_search([1,2,4]),
            "inference_memory": tune.grid_search([16]),
            "inference_batch": tune.grid_search([1, 2, 4, 8, 16, 32, 64])
        },
        verbose=1,
        scheduler=hyperband,
        fail_fast=True)
 
    tuning_duration = time.time() - tuning_start
    print("Tuning duration: %d" % tuning_duration)

    trials = analysis.trials
    for trial in trials:
        print("%d,%d,%d,%d,%f" % 
                (trial.config['inference_cores'],
                trial.config['inference_memory'],
                trial.config['n'],
                trial.config['inference_batch'],
                trial.metric_analysis['inference_duration']['avg']))
        #print(trial.config)
        #print("inference_duration: " + str(trial.metric_analysis['inference_duration']))

    best_config_acc = analysis.get_best_config(metric="inference_duration", mode="min")
    print("Best parameters found were: ", best_config_acc)
