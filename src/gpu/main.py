from tuning import EdgeTuneV1
from tuning import EdgeTuneV2
from tuning import EdgeTuneV3
#from tuning import EdgeTuneGrid
from tuning import InferenceServer

if __name__ == "__main__":
    #for n in [3]:#, 5, 7, 9, 18, 27]:
    #    EdgeTuneGrid.runSearch(n)
    #EdgeTuneV1.runSearch()
    #EdgeTuneV2.runSearch()
    EdgeTuneV3.runSearch()
    #InferenceServer.runSearch(3, result)
