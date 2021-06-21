import lib.rapl.rapl as rapl
from tuning import TuneV3
import time

if __name__ == "__main__":
    tuning_start = time.time()
    start_energy = rapl.RAPLMonitor.sample()

    TuneV3.runSearch()

    tuning_duration = time.time() - tuning_start
    end_energy = rapl.RAPLMonitor.sample()
    diff = end_energy-start_energy
    tuning_energy = diff.energy('package-0')
    print("Tuning duration: %d\nTuning energy: %f" % (tuning_duration, tuning_energy))
    #InferenceServer.runSearch(3, result)
