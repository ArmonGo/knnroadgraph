import logging

import networkx as nx
import numpy as np
from shapely import ops
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point
from sklearn.preprocessing import MinMaxScaler


def X_mask_to_target_node_mask(graph, mask):
    attr_dict = {node: attrs for node, attrs in graph.graph.nodes(data=True)}
    type_mask = np.array(
        [attr_dict[i]["type"] == "house" for i in range(len(graph.graph.nodes))]
    )
    house_idx = np.array(range(len(graph.graph.nodes)))[type_mask]
    return house_idx[mask]


class GeoGraph:
    def __init__(
        self,
        roads,
        houses,
        y_vals,
        simplify=100,
        preserve_topology=True,
    ):
        assert all(type(house) == Point for house in houses)
        assert type(roads) == MultiLineString
        assert len(houses) == len(y_vals)

        self.roads = roads
        self.houses = houses
        self.X_vals = np.array(
            [[house.coords[0][0], house.coords[0][1]] for house in houses]
        )
        self.y_vals = np.array(y_vals)
        self.simplify = simplify
        self.preserve_topology = preserve_topology
        self.graph = None
        self._process_geometry()
        self._construct_graph()
        self.scaler = MinMaxScaler().fit(self.X_vals)

    def get_X(self, mask, scaled=True):
        if scaled:
            return self.scaler.transform(self.X_vals[mask, :])
        return self.X_vals[mask, :]

    def get_y(self, mask, log1p=False):
        if log1p:
            return np.log1p(self.y_vals[mask])
        return self.y_vals[mask]

    def _process_geometry(self):
        logging.debug("Merging geometry...")
        self.roads = ops.linemerge(MultiLineString(self.roads))
        if self.simplify:
            logging.debug("Simplifying geometry...")
            self.roads = self.roads.simplify(self.simplify, self.preserve_topology)
        logging.debug("Calculating connection points...")
        connection_points = [
            ops.nearest_points(self.roads, house)[0] for house in self.houses
        ]
        logging.debug("Splitting geometry...")
        split_geometry = ops.split(self.roads, MultiPoint(connection_points))
        logging.debug("Unionizing geometry")
        self.roads = MultiLineString(split_geometry).union(
            MultiLineString(
                [LineString([h, cp]) for h, cp in zip(self.houses, connection_points)]
            )
        )

    def _construct_graph(self):
        point_to_node_idx = {}
        edges = []
        nodes = []

        def get_or_extend(p, d):
            if p in d:
                return d[p], False
            d[p] = len(d)
            return d[p], True

        houses = []
        for house, y_val in zip(self.houses, self.y_vals):
            ni, new_i = get_or_extend(house.coords[0], point_to_node_idx)
            nodes.append(
                (
                    ni,
                    {
                        "type": "house",
                        "x": house.coords[0][0],
                        "y": house.coords[0][1],
                        "target": y_val,
                    },
                )
            )
            houses.append(ni)

        for line in self.roads.geoms:
            for i in range(len(line.coords) - 1):
                dist = np.linalg.norm(
                    np.array(line.coords[i]) - np.array(line.coords[i + 1])
                )
                ni, new_i = get_or_extend(line.coords[i], point_to_node_idx)
                nj, new_j = get_or_extend(line.coords[i + 1], point_to_node_idx)
                if ni in houses or nj in houses:
                    dist = 0
                edges.append((ni, nj, dist))
                if new_i:
                    nodes.append(
                        (
                            ni,
                            {
                                "type": "road",
                                "x": line.coords[i][0],
                                "y": line.coords[i][1],
                            },
                        )
                    )
                if new_j:
                    nodes.append(
                        (
                            nj,
                            {
                                "type": "road",
                                "x": line.coords[i + 1][0],
                                "y": line.coords[i + 1][1],
                            },
                        )
                    )
        self.graph = nx.Graph()
        self.graph.add_weighted_edges_from(edges)
        self.graph.add_nodes_from(nodes)
