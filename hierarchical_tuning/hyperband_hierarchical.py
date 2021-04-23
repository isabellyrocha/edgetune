#!/usr/bin/env python

import subprocess
import argparse
import json
import os
import time

import numpy as np

import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

class MyTrainableClass(tune.Trainable):

    def setup(self, config):
        self.timestep = 0
    
    def set_cores(self, cores):
        command = "ps -x | grep hyperband_hierarchical | awk '{print $1}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
        subprocess.Popen(["ssh", "eiger-1.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Changed number of cores to %s... " % cores)

    def step(self):
        self.timestep += 1
        #v = np.tanh(float(self.timestep) / self.config.get("width", 1))
        #v *= self.config.get("height", 1)
        #time.sleep(0.1)
        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

        # Here we use `episode_reward_mean`, but you can also report other
        # objectives such as loss or accuracy.
        inputs = keras.Input(shape=(784,), name="digits")
        x1 = layers.Dense(64, activation="relu")(inputs)
        x2 = layers.Dense(64, activation="relu")(x1)
        outputs = layers.Dense(10, name="predictions")(x2)
        model = keras.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.SGD(learning_rate=1e-3)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        
        self.set_cores(self.config.get("cores", 4))
        batch_size = self.config.get("batch", 128)
        print("Using bach size %d..." % self.config.get("batch", 128))
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_batch_size = 64
        val_dataset = val_dataset.batch(val_batch_size)


        epochs = 1
        fwd_time = 0
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            epoch_tim = time.time()
            fwd_time = 0
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                start = time.time()
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)  # Logits for this minibatch
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                elapsed = time.time() - start
                fwd_time += elapsed
                #print(elapsed)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                train_acc_metric.update_state(y_batch_train, logits)
                if step % 200 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))
            print(train_acc_metric.result())
            print("Forward: %f\n" % fwd_time)
            print("Epoch: %f\n" % (time.time() - epoch_tim))

        
        train_acc = train_acc_metric.result()
        ratio = float(train_acc)/fwd_time

        train_acc_metric.reset_states()
        start = time.time()
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        print(val_acc)
        val_acc_metric.reset_states()
        elapsed = time.time() - start
        print("Inference: %f\n" % elapsed)
        v = float(train_acc)/fwd_time
        print("RAAAATIOOOOOOO" +str(v))
        return {"ratio": ratio, "accuracy": float(train_acc), "duration": fwd_time}

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
    #os.system("taskset -p -c 0,1 %d" % os.getpid())
    ray.init(num_cpus=8)

    begin_tuning = time.time()

    #tf.config.threading.set_inter_op_parallelism_threads(2)
    #tf.config.threading.set_intra_op_parallelism_threads(2)
    # Hyperband early stopping, configured with `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    hyperband = HyperBandScheduler(time_attr="training_iteration", metric="accuracy", mode = "max", max_t=18)

    analysis = tune.run(
        MyTrainableClass,
        name="hyperband_test",
        num_samples=1, #if args.smoke_test else 200,
        #metric="episode_reward_mean",
        #mode="max",
        stop={"training_iteration": 2},
        resources_per_trial={
            "cpu": 8,
            "gpu": 0
        },
        config={
            #"cores": tune.grid_search([1, 2, 4])
            "batch": tune.grid_search([32, 64, 128])
        },
        verbose=1,
        scheduler=hyperband,
        fail_fast=True)
 
    trials = analysis.trials
    for trial in trials:
        print(trial.config)
        print (trial.metric_analysis['ratio'])
    
    best_batch = analysis.get_best_config(
    metric="accuracy", mode="max")

    print("Best hyperparameters found were: ", best_batch)


    hyperband_phase_2 = HyperBandScheduler(time_attr="training_iteration", metric="duration", mode = "min", max_t=18)

    analysis = tune.run(
        MyTrainableClass,
        name="hyperband_test",
        num_samples=1, #if args.smoke_test else 200,
        #metric="episode_reward_mean",
        #mode="max",
        stop={"training_iteration": 2},
        resources_per_trial={
            "cpu": 8,
            "gpu": 0
        },
        config={
            "cores": tune.grid_search([1, 2, 4]),
            "batch": tune.grid_search([best_batch['batch']])
        },
        verbose=1,
        scheduler=hyperband_phase_2,
        fail_fast=True)

    trials = analysis.trials
    for trial in trials:
        print(trial.config)
        print (trial.metric_analysis['duration'])
    print("Best hyperparameters found were: ", analysis.get_best_config(
    metric="duration", mode="min"))

    end_tuning = time.time()
    print("Tuning time: %d" % (end_tuning-begin_tuning))
