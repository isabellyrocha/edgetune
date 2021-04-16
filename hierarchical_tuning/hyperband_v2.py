#!/usr/bin/env python

import argparse
import json
import os
import time

import numpy as np

import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class MyTrainableClass(tune.Trainable):
    """Example agent whose learning curve is a random sigmoid.
    The dummy hyperparameters "width" and "height" determine the slope and
    maximum reward value reached.
    """

    def setup(self, config):
        self.timestep = 0

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    def step(self):
        self.timestep += 1
        v = np.tanh(float(self.timestep) / self.config.get("width", 1))
        v *= self.config.get("height", 1)
        time.sleep(0.1)

#        tf.config.threading.set_inter_op_parallelism_threads(4)
#        tf.config.threading.set_intra_op_parallelism_threads(4)

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

        batch_size = 32
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
        val_dataset = val_dataset.batch(batch_size)


        epochs = 3
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
            print("Forward: %f\n" % fwd_time)
            print("Epoch: %f\n" % (time.time() - epoch_tim))

        train_acc = train_acc_metric.result()
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
        return {"episode_reward_mean": v}

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
    ray.init(num_cpus=1)

    #tf.config.threading.set_inter_op_parallelism_threads(2)
    #tf.config.threading.set_intra_op_parallelism_threads(2)
    # Hyperband early stopping, configured with `episode_reward_mean` as the
    # objective and `training_iteration` as the time unit,
    # which is automatically filled by Tune.
    hyperband = HyperBandScheduler(time_attr="training_iteration", max_t=1)

    analysis = tune.run(
        MyTrainableClass,
        name="hyperband_test",
        num_samples=1, #if args.smoke_test else 200,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": 1},
        config={
            "width": tune.randint(10, 90),
            "height": tune.randint(0, 100)
        },
        verbose=1,
        scheduler=hyperband,
        fail_fast=True)
 
    print("Best hyperparameters found were: ", analysis.best_config)
