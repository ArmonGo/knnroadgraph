import matplotlib as mpl
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def colormap(colors=["#1098f7", "#fb3640"], name="custom"):
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, np.array(colors))
    return cmap


def plot_segment_predictions(
    lines,
    prices,
    cmap=None,
    norm=plt.Normalize(),
    linewidth=1,
    alpha=1.0,
    xlim=(140000, 159030),
    ylim=(203000, 229775),
):
    if cmap is None:
        cmap = colormap()
    ax = plt.gca()
    lc = mcoll.LineCollection(
        lines, array=prices, cmap=cmap, norm=norm, linewidth=linewidth, alpha=alpha
    )
    if xlim:
        ax.set(xlim=xlim)
    if ylim:
        ax.set(ylim=ylim)
    ax.add_collection(lc)
    return lc


def to_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments.tolist()


def get_segment_predictions(G, predictions, steps=10):
    edges = list(G.edges)
    loc_x = nx.get_node_attributes(G, "x")
    loc_y = nx.get_node_attributes(G, "y")
    distance = nx.get_edge_attributes(G, "weight")

    segment = []
    z_list = []

    for node_1, node_2 in edges:
        x = np.array([loc_x[node_1], loc_x[node_2]])
        y = np.array([loc_y[node_1], loc_y[node_2]])
        z = np.array([predictions[node_1], predictions[node_2]])
        z = np.log(z)
        step = int(distance[(node_1, node_2)] / steps)
        path = mpath.Path(np.column_stack([x, y]))
        verts = path.interpolated(steps=step).vertices

        x, y = verts[:, 0], verts[:, 1]
        z = np.linspace(z[0], z[1], step)
        segment = segment + to_segments(x, y)
        z_list = z_list + z.tolist()

    return segment, z_list
