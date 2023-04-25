import matplotlib.pyplot as plt
import networkx as nx


def plot_geograph(graph, xlim=None, ylim=None, y_vals=None):
    for line in graph.roads.geoms:
        plt.plot(*line.xy, lw=0.5, c="#555555aa")
    if graph.houses:
        plt.scatter(
            [house.x for house in graph.houses],
            [house.y for house in graph.houses],
            s=10,
            c=y_vals
            if y_vals is not None
            else ("blue" if not len(graph.y_vals) else graph.y_vals),
        )
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if not xlim and not ylim:
        plt.axis("equal")
    plt.show()


def plot_nxgraph(G, xlim=None, ylim=None, use_pos=True):
    fixed_pos = {node: (attrs["x"], attrs["y"]) for node, attrs in G.nodes(data=True)}
    pos = nx.spring_layout(G, pos=fixed_pos, fixed=fixed_pos.keys())
    edge_colors = [G[u][v]["weight"] for u, v in G.edges()]
    color_map = plt.cm.winter
    node_sizes = [
        10 if att["type"] == "house" else 0 for node, att in G.nodes(data=True)
    ]
    node_colors = [
        att["target"] if att["type"] == "house" else 0
        for node, att in G.nodes(data=True)
    ]
    if use_pos:
        nx.draw_networkx(
            G,
            pos=pos,
            with_labels=False,
            alpha=0.5,
            node_size=node_sizes,
            node_color=node_colors,
            edge_color=edge_colors,
            edge_cmap=color_map,
        )
    else:
        nx.draw_networkx(
            G,
            with_labels=False,
            alpha=0.5,
            node_size=node_sizes,
            node_color=node_colors,
            edge_color=edge_colors,
            edge_cmap=color_map,
        )
    if xlim:
        plt.xlim(*xlim)
    if ylim:
        plt.ylim(*ylim)
    if not xlim and not ylim:
        plt.axis("equal")
    plt.show()
