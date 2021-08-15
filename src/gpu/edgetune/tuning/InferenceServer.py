#from workloads.MyTrainableClass import MyTrainableClass
from workloads.ic.Inference import Inference
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune import CLIReporter
from ray.tune.suggest.bohb import TuneBOHB
from ray import tune
import json
import ray
import os

def runSearch(embed_dim, result):
    import ConfigSpace as CS  # noqa: F401

    config={
            "iterations": 1,
            "embed_dim": embed_dim,
            #"n": n,
            "inference_cores": tune.choice([1, 2, 4]),
            "inference_batch": tune.choice([1, 10, 100])
    }

    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1,
        reduction_factor=2
    )

    bohb_search = TuneBOHB(max_concurrent=1)

#    reporter = CLIReporter(max_progress_rows=50)


    analysis = tune.run(
        Inference,
        name="InferenceServer[BOHB]",
        config=config,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        num_samples=4,
        stop={"training_iteration": 1},
        metric="inference_duration",
        mode="min",
        resources_per_trial={
            "cpu": 4,
            "gpu": 0
        }
        #progress_reporter=reporter  
    )

    for key in analysis.best_result.keys():
        result[key] = analysis.best_result[key]
    
    return result
