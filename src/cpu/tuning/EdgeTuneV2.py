from workloads.cifar10resnet.DatasetTraining import DatasetTraining
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune import CLIReporter
from ray import tune
import json
import ray
import os

def runSearch():
    import ConfigSpace as CS  # noqa: F401

    ray.init(num_cpus=8)

    config={
            "iterations": 100,
            "n": tune.choice([3, 5, 7]),
            "train_batch": tune.choice([32, 64, 128, 256, 512])
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=2)

    bohb_search = TuneBOHB(max_concurrent=1)

    reporter = CLIReporter(max_progress_rows=50)
    reporter.add_metric_column("training_accuracy")
    reporter.add_metric_column("training_duration")
    reporter.add_metric_column("inference_duration")
    reporter.add_metric_column("runtime_ratio")
    reporter.add_metric_column("inference_cores")
    reporter.add_metric_column("inference_batch")

    analysis = tune.run(
        DatasetTraining,
        name="EdgeTuneV1[BOHB]",
        config=config,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=10,
        stop={"training_iteration": 100},
        metric="runtime_ratio",
        mode="min",
        resources_per_trial={
            "cpu": 4,
            "gpu": 0
        },
        progress_reporter=reporter)

    print("Best hyperparameters found were: ", analysis.best_config)
