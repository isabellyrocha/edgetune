from tuning import InferenceServer

if __name__ == "__main__":
    for cores in [1]:
        for n in [3, 5, 7, 9, 18, 27]:
            result = {}
            InferenceServer.runSearch(n, cores, result)
            print(result)
