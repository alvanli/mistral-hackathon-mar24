from text_clustering import ClusterClassifier

if __name__ == "__main__":
    # cc = ClusterClassifier()
    # embs, labels, summaries = cc.fit()
    # cc.save("./last_1000")

    cc = ClusterClassifier()
    cc.load("./last_1000")
    cc.show(interactive=True)
