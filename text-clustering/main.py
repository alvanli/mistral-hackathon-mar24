from text_clustering import ClusterClassifier

if __name__ == "__main__":
    cc = ClusterClassifier()
    embs, labels, summaries = cc.fit()
    cc.save("./cc_100k")

    cc = ClusterClassifier()
    cc.load("./cc_100k")
    # visualize
    cc.show()
