#from workloads.MyTrainableClass import MyTrainableClass
from workloads.ResnetCifar10Inf import ResnetCifar10Inf
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from ray import tune
import json
import ray
import os

def runSearch(n):
    import ConfigSpace as CS  # noqa: F401

    ray.init(num_cpus=8)

    config={
            "iterations": 1,
            "n": n,
            "inference_cores": tune.choice([1, 2, 4]),
            "inference_batch": tune.choice([1, 2, 4, 8, 16, 32, 64])
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=100,
        reduction_factor=2)

    bohb_search = TuneBOHB(max_concurrent=1)

    analysis = tune.run(
        ResnetCifar10,
        name="EdgeTuneV1[BOHB]",
        config=config,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=8,
        stop={"training_iteration": 100},
        metric="training_duration",
        mode="max",
        resources_per_trial={
            "cpu": 8,
            "gpu": 0
        })

    print("Best hyperparameters found were: ", analysis.best_config)
