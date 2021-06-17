from ray.tune.schedulers import HyperBandScheduler, AsyncHyperBandScheduler
from workloads.cifar10resnet.EpochTraining import EpochTraining
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune import CLIReporter
from ray import tune
import json
import ray
import os

def runSearch():
    import ConfigSpace as CS  # noqa: F401

    ray.init(num_cpus=4, num_gpus=8)

    config={
            "iterations": 200,
            "n": tune.grid_search([3, 5, 7, 9, 18, 27]),
            "train_batch": tune.choice([32])
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=200,
        reduction_factor=2)

    bohb_search = TuneBOHB(max_concurrent=1)

    reporter = CLIReporter(max_progress_rows=50)
    reporter.add_metric_column("epochs")
    reporter.add_metric_column("training_accuracy")
    reporter.add_metric_column("training_duration")
    reporter.add_metric_column("inference_duration")
    reporter.add_metric_column("runtime_ratio")
    reporter.add_metric_column("inference_cores")
    reporter.add_metric_column("inference_batch")
    '''
    analysis = tune.run(
        EpochTraining,
        name="EdgeTuneV1[BOHB]",
        config=config,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=6,
        stop={"epochs": 200},
        metric="runtime_ratio",
        mode="min",
        resources_per_trial={
            "cpu": 1,
            "gpu": 8
        },
        progress_reporter=reporter)
    '''
    hyperband = HyperBandScheduler(time_attr="epochs", metric="runtime_ratio", mode = "max", max_t=6)

    analysis = tune.run(
        EpochTraining,
        name="EdgeTuneV1[GridSearch]",
        num_samples=1,
        stop={"epochs": 200},
        resources_per_trial={
            "cpu": 2,
            "gpu": 8
        },
        config=config,
        verbose=1,
        scheduler=hyperband,
        fail_fast=True,
        progress_reporter=reporter)

    print("Best hyperparameters found were: ", analysis.best_config)
