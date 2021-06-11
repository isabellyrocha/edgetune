from tuning import InferenceServer

if __name__ == "__main__":
    result = {}
    InferenceServer.runSearch(3, result)
    print(result)
