import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from .base import BaseAlgorithm


class SVM(BaseAlgorithm):
    def param_names(self):
        return ["kernel", "C", "epsilon"]

    def train(self, graph, mask):
        self.svm = SVR(
            kernel=self.params["kernel"],
            C=self.params["C"],
            epsilon=self.params["epsilon"],
        )
        self.svm.fit(graph.get_X(mask), graph.get_y(mask))

    def predict(self, graph, mask):
        return self.svm.predict(graph.get_X(mask))


class Kmeans(BaseAlgorithm):
    def param_names(self):
        return ["k", "seed"]

    def train(self, graph, mask):
        with threadpool_limits(limits=3):
            self.kmeans = KMeans(
                n_clusters=min(self.params["k"], len(mask)),
                n_init="auto",
                random_state=self.params["seed"],
            )
            self.kmeans.fit(graph.get_X(mask))
        self.cluster_labels = self.kmeans.labels_
        self.cluster_centroids = self.kmeans.cluster_centers_
        self.cluster_prices = []
        for i in range(self.params["k"]):
            node_ix = np.argwhere(self.cluster_labels == i)
            train_y = graph.get_y(mask)[node_ix]
            self.cluster_prices.append(np.mean(train_y))

    def predict(self, graph, mask):
        labels = self.kmeans.predict(graph.get_X(mask))
        return np.array([self.cluster_prices[label] for label in labels])


class DecisionTree(BaseAlgorithm):
    def param_names(self):
        return ["ccp_alpha", "seed"]

    def train(self, graph, mask):
        self.dtree = DecisionTreeRegressor(
            ccp_alpha=self.params["ccp_alpha"],
            random_state=self.params["seed"],
        )
        self.dtree.fit(graph.get_X(mask), graph.get_y(mask))

    def predict(self, graph, mask):
        return self.dtree.predict(graph.get_X(mask))


class RandomForest(BaseAlgorithm):
    def param_names(self):
        return ["max_features", "seed"]

    def train(self, graph, mask):
        self.rf = RandomForestRegressor(
            max_features=self.params["max_features"],
            random_state=self.params["seed"],
        )
        self.rf.fit(graph.get_X(mask), graph.get_y(mask))

    def predict(self, graph, mask):
        return self.rf.predict(graph.get_X(mask))


class KNN(BaseAlgorithm):
    def param_names(self):
        return ["n_neighbors", "weights"]

    def train(self, graph, mask):
        self.knn = KNeighborsRegressor(
            n_neighbors=min(self.params["n_neighbors"], len(mask)),
            weights=self.params["weights"],
        )
        self.knn.fit(graph.get_X(mask), graph.get_y(mask))

    def predict(self, graph, mask):
        return self.knn.predict(graph.get_X(mask))
