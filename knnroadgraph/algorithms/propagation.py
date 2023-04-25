import networkx as nx
import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
from threadpoolctl import threadpool_limits
from torch_geometric.nn import LabelPropagation as TorchLabelPropagation
from torch_geometric.utils import from_networkx, index_to_mask

from .base import BaseAlgorithm
from .geograph import X_mask_to_target_node_mask


def geograph_to_pytorch_graph(geograph, train_mask):
    attrs = {}
    type_attr = nx.get_node_attributes(geograph.graph, "type")
    for node in nx.get_node_attributes(geograph.graph, "type"):
        if type_attr[node] == "road":
            attrs[node] = {"target": -1}
    nx.set_node_attributes(geograph.graph, attrs)

    g = from_networkx(
        geograph.graph,
        group_node_attrs=["x", "y", "target"],
        group_edge_attrs=["weight"],
    )

    g.y = g.x[:, -1]
    g.x = g.x[:, :2]
    type_mask = torch.tensor([i != "road" for i in g.type])
    node_num = g.x.shape[0]
    g.node_house_num = type_mask.sum()
    g.type_mask = type_mask

    g.train_mask_i = torch.from_numpy(
        X_mask_to_target_node_mask(geograph, np.array(train_mask))
    )
    g.train_mask = index_to_mask(g.train_mask_i, size=node_num)

    return g


class LabelPropagation(BaseAlgorithm):
    def param_names(self):
        return ["num_intervals", "num_layers", "alpha", "bin_strategy"]

    def train(self, graph, mask):
        self.g = geograph_to_pytorch_graph(graph, mask)
        self.scaled_weights = torch.exp(
            -(self.g.edge_attr - self.g.edge_attr.min())
            / (self.g.edge_attr.max() - self.g.edge_attr.min())
        )
        with threadpool_limits(limits=1):
            self.binner = KBinsDiscretizer(
                n_bins=min(self.params["num_intervals"], len(mask)),
                encode="ordinal",
                strategy=self.params["bin_strategy"],
            )
            self.y_cats = self.binner.fit_transform(
                graph.get_y(mask).reshape(-1, 1)
            ).squeeze()
        self.cat_avgs = np.array(
            [
                graph.get_y(mask)[self.y_cats == i].mean()
                if len(graph.get_y(mask)[self.y_cats == i])
                else np.nan
                for i in range(self.params["num_intervals"])
            ]
        )
        self.cat_avgs = np.nan_to_num(self.cat_avgs, nan=np.mean(graph.get_y(mask)))
        self.y_cats_t = torch.tensor(
            self.binner.transform(self.g.y.reshape(-1, 1))
        ).view(-1, 1)
        self.model = TorchLabelPropagation(
            num_layers=self.params["num_layers"], alpha=self.params["alpha"]
        )
        self.out = self.model(
            self.y_cats_t,
            self.g.edge_index,
            mask=self.g.train_mask,
            edge_weight=self.scaled_weights,
        )

    def predict(self, graph, mask):
        _, pred = self.out.max(dim=1)
        return self.cat_avgs[pred.numpy()[X_mask_to_target_node_mask(graph, mask)]]
