import lib.rapl.rapl as rapl
#from tuning import EdgeTuneV1
from tuning import EdgeTuneV2
#from tuning import EdgeTuneV3
#from tuning import EdgeTuneGrid
from tuning import InferenceServer
import time

if __name__ == "__main__":
    tuning_start = time.time()
    start_energy = rapl.RAPLMonitor.sample()
    
    #EdgeTuneV1.runSearch()
    EdgeTuneV2.runSearch()
    #EdgeTuneV3.runSearch("runtime_ratio")    

    tuning_duration = time.time() - tuning_start
    end_energy = rapl.RAPLMonitor.sample()
    diff = end_energy-start_energy
    tuning_energy = diff.energy('package-0')
    print("Tuning duration: %d\nTuning energy: %f" % (tuning_duration, tuning_energy))
