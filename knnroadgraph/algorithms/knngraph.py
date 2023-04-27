import networkx as nx
import numpy as np
from scipy.sparse.csgraph import csgraph_from_dense
from sklearn.neighbors import KNeighborsRegressor, sort_graph_by_row_values

from .base import BaseAlgorithm
from .geograph import X_mask_to_target_node_mask


class SkKnnGraph(BaseAlgorithm):
    def param_names(self):
        return [
            "distance_matrix_file",
            "n_neighbors",
            "distance_scaler",
            "distance_minmax",
            "weight_method",
            "temperature",
            "power",
            "decay",
            "jitter",
        ]

    def construct_distance_matrix(self, graph, file_name):
        if not hasattr(graph, "_dist_matrix"):
            graph._dist_matrix = {}
        if file_name not in graph._dist_matrix:
            dist = np.full(
                (graph.graph.number_of_nodes(), graph.graph.number_of_nodes()), -1
            )
            for i in self.train_node_ix:
                for j, dis in nx.shortest_path_length(
                    graph.graph, target=i, weight="weight"
                ).items():
                    dist[i][j] = dis
                    dist[j][i] = dis
            dist = csgraph_from_dense(dist, null_value=-1)
            graph._dist_matrix[file_name] = sort_graph_by_row_values(
                dist, copy=True, warn_when_not_sorted=False
            )
        return graph._dist_matrix[file_name]

    def construct_knn_model(self, graph, file_name):
        if not hasattr(graph, "_knn_models"):
            graph._knn_models = {}
        if file_name not in graph._knn_models:
            weights = self.wfunc
            if self.params["weight_method"] in ["uniform", "distance"]:
                weights = self.params["weight_method"]
            knn = KNeighborsRegressor(
                n_neighbors=min(self.params["n_neighbors"], len(self.train_mask)),
                weights=weights,
                metric="precomputed",
            )
            targets = nx.get_node_attributes(graph.graph, "target")
            targets = np.array(
                [
                    targets.get(i, self.avg_price)
                    for i in range(graph.graph.number_of_nodes())
                ]
            )
            knn.fit(self.distance_matrix, targets)
            graph._knn_models[file_name] = knn
        return graph._knn_models[file_name]

    def train(self, graph, mask):
        self.graph = graph
        self.train_mask = mask
        self.train_node_ix = X_mask_to_target_node_mask(graph, np.array(mask))
        self.avg_price = np.mean(graph.get_y(mask))
        self.distance_matrix = self.construct_distance_matrix(
            graph, self.params["distance_matrix_file"]
        )
        self.knn = self.construct_knn_model(
            graph, f"{self.params['distance_matrix_file']}.knn"
        )
        self.knn.n_neighbors = min(self.params["n_neighbors"], len(mask))
        fallback_weights = "distance"
        if self.params["weight_method"] in ["uniform", "distance"]:
            fallback_weights = self.params["weight_method"]
        self.knn_fallback = KNeighborsRegressor(
            n_neighbors=min(self.params["n_neighbors"], len(mask)),
            weights=fallback_weights,
        )
        self.knn_fallback.fit(graph.get_X(mask), graph.get_y(mask))

    def wfunc(self, dists):
        if self.params["distance_scaler"] == "normalized":
            mu = np.mean(dists)
            std = np.std(dists)
            dists_s = (dists - mu) / std
        elif self.params["distance_scaler"] == "decay":
            dists_s = 1 - np.exp(-self.params["decay"] * dists)
        else:
            dists_s = dists

        if np.sum(dists_s) == 0:
            nearest_percentage = np.ones((len(dists),)) / len(dists)
        else:
            if self.params["distance_minmax"]:
                r = np.max(dists_s) - np.min(dists_s)
                if r == 0:
                    dists_s = np.ones(dists_s.shape)
                else:
                    dists_s = (dists_s - np.min(dists_s)) / (
                        np.max(dists_s) - np.min(dists_s)
                    )
            if self.params["weight_method"] == "softmax":
                dists_s = np.exp(-dists_s / self.params["temperature"])
            elif self.params["weight_method"] == "argmax":
                eps = 0.0000001
                dists_s = ((eps + np.max(dists_s)) / (eps + dists_s)) ** self.params[
                    "power"
                ]

            nearest_percentage = dists_s / np.sum(dists_s)

        if self.params["jitter"]:
            nearest_percentage += (
                np.random.random(nearest_percentage.shape) - 0.5 * self.params["jitter"]
            )

        return nearest_percentage

    def predict_node_fallback(self, graph, node_ix):
        or_n = self.knn.n_neighbors
        pred = None
        # Scikit-learn complains if not the full set of neighbors are connected
        # Try with descending values, if that fails, revert to euclidean
        for r in range(or_n, 0, -1):
            try:
                self.knn.n_neighbors = r
                pred = self.knn.predict(self.distance_matrix[node_ix])[0]
                break
            except ValueError:
                pass
        if pred is None:
            try:
                # For this to work we need the x y coords
                # as this might be a road node
                x = graph.graph.nodes[node_ix]["x"]
                y = graph.graph.nodes[node_ix]["y"]
                scaled = graph.scaler.transform(np.array([x, y]).reshape(1, -1))
                pred = self.knn_fallback.predict(scaled)[0]
            except ValueError:
                pred = self.avg_price
        self.knn.n_neighbors = or_n
        return pred

    def predict_nodes(self, graph, node_ixs):
        try:
            return self.knn.predict(self.distance_matrix[node_ixs])
        except ValueError:
            preds = []
            for node_ix in node_ixs:
                try:
                    preds.append(self.knn.predict(self.distance_matrix[node_ix])[0])
                except ValueError:
                    preds.append(self.predict_node_fallback(graph, node_ix))
            return np.array(preds)

    def predict_node(self, graph, node_ix):
        try:
            return self.knn.predict(self.distance_matrix[node_ix])[0]
        except ValueError:
            return self.predict_node_fallback(graph, node_ix)

    def predict(self, graph, mask):
        node_ix = X_mask_to_target_node_mask(graph, mask)
        return self.predict_nodes(graph, node_ix)
