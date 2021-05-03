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
        command = "ps -x | grep hyperband_onefold | awk '{print $1}' | while read line ; do sudo taskset -cp -pa 0-%d $line; done" % (int(cores)-1)
        subprocess.Popen(["ssh", "eiger-1.maas", command], shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("Changed number of cores to %s... " % cores)

    def step(self):
        self.timestep += 1

        tf.config.threading.set_inter_op_parallelism_threads(8)
        tf.config.threading.set_intra_op_parallelism_threads(8)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        fwd_time = 0
        pbar = tqdm(total=args.time)

        epochs = 1
        for epoch in range(epoch):
            print("Start of epoch %d" % epoch)
            epoch_start = time.time()

            aggregated_forward_duration = 0
            for (images, target) in loader:
                start = time.time()

                print("forward")  # workloads should print the phase they are in when starting them
                out = model(images)
                loss = criterion(out, target)

                elapsed = time.time() - start
                aggregated_forward_duration += elapsed
                pbar.update(elapsed)

                print("backward")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                #if fwd_time > args.time:
                #    meter.end()
                #    return meter.result


        return {"training_accuracy": training_accuracy, "inference_accuracy": inference_accuracy, "forward_duration": forward_duration, "epoch_duration": epoch_duration, "inference_duration": inference_duration, "proxy_ratio": proxy_ratio, "real_ratio": real_ratio}

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
    
    ray.init(num_cpus=8)


    tuning_start = time.time()

    hyperband = HyperBandScheduler(time_attr="training_ratio", metric="proxy_ratio", mode = "max", max_t=18)

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
            "cores": tune.grid_search([4]),
            "batch": tune.grid_search([32])
        },
        verbose=1,
        scheduler=hyperband,
        fail_fast=True)
 
    trials = analysis.trials
    for trial in trials:
        print(trial.config)
        print(trial.metric_analysis['proxy_ratio'])
        print(trial.metric_analysis['real_ratio'])

    best_config = analysis.get_best_config(metric="proxy_ratio", mode="max")
    print("Best hyperparameters found were: ", best_config)

    tuning_duration = time.time() - tuning_start
    print("Tuning duration: %d" % tuning_duration)
