import glob
import itertools as it

import click
import numpy as np
import pandas as pd
import pymincut.pygraph as pg

"""
These statistics functions all take in a
pandas dataframe of the nodes and the edges
"""


def size(nodes, _):
    return len(nodes)


def interior(nodes, edges):
    return sum(edges["u"].isin(nodes) & edges["v"].isin(nodes))


def boundary(nodes, edges):
    return sum(edges["u"].isin(nodes) ^ edges["v"].isin(nodes))


def conductance(nodes, edges):
    interior = sum(edges["u"].isin(nodes) & edges["v"].isin(nodes))
    boundary = sum(edges["u"].isin(nodes) ^ edges["v"].isin(nodes))
    return boundary / (2 * interior + boundary) if boundary > 0 else 0.0


def density(nodes, edges):
    if len(nodes) == 1:
        return 0
    interior = sum(edges["u"].isin(nodes) & edges["v"].isin(nodes))
    return 2 * interior / (len(nodes) * (len(nodes) - 1))


def modularity(nodes, edges):
    interior = sum(edges["u"].isin(nodes) & edges["v"].isin(nodes))
    boundary = sum(edges["u"].isin(nodes) ^ edges["v"].isin(nodes))
    total = len(edges)
    return (interior / total) - ((2 * interior + boundary) / (2 * total)) ** 2


def connectivity(nodes, edges):
    interior = edges["u"].isin(nodes) & edges["v"].isin(nodes)
    interior = edges[interior].itertuples(index=False, name=None)
    graph = pg.PyGraph(list(nodes), list(interior))
    return graph.mincut("noi", "bqueue", False)[2]


def mcd(nodes, edges):
    interior = edges[edges["u"].isin(nodes) & edges["v"].isin(nodes)]
    return pd.concat([interior["u"], interior["v"]]).value_counts().min()


def jaccard(ikc_com, csk_com):
    i = np.intersect1d(ikc_com, csk_com)
    u = np.union1d(ikc_com, csk_com)
    return len(i) / len(u)


def generate_row(ikc_com, csk_com, node, edges, data, ikc_com_id):
    """
    node,jaccard,ikc_size,ikc_conductance,ikc_density,ikc_modularity,ikc_connectivity,csk_size,csk_conductance,csk_density,csk_modularity,csk_connectivity
    """
    fs = [size, interior, boundary, conductance, density, modularity, connectivity, mcd]
    ds = [ikc_com, csk_com]
    row = [f(nodes, edges) for nodes, f in it.product(ds, fs)]
    jc = jaccard(ikc_com, csk_com)
    data.append([node, jc, ikc_com_id] + row)


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--ikc_com", required=True, type=click.Path(exists=True))
@click.option("--csk_com", required=True, type=click.Path(exists=True))
@click.option("--csk_id", required=True, type=click.INT)
@click.option("--output", required=True)
def main(edgelist, ikc_com, csk_com, csk_id, output):
    """
    csk_com: directory of all the community search-ed communities
    ikc_com: file of the clustering given by IKC
    csk_id: the filename that the csk clustering must end in
    output:
        where the dataframe with all the community search nodes/stats is written to
    """

    edges = pd.read_csv(edgelist, sep="\\s+", names=["u", "v"])
    data = []

    ikc_all = pd.read_csv(
        ikc_com,
        usecols=[0, 1],  # type: ignore[arg-type]
        sep=",",
        index_col=0,
        names=["id", "c", "0", "1"],
    )
    for kcore in glob.glob(f"{csk_com}/*/*"):
        if csk_id != -1 and kcore.split("/")[-1] != f"kcore_k{csk_id}.txt":
            continue

        node = int(kcore.split("/")[-2])
        print("processing:", node)

        # get the ikc found community
        ikc_com_id = ikc_all["c"].loc[node]
        ikc_nodes = ikc_all[ikc_all["c"] == ikc_com_id].index

        # get the community searched community
        csk = pd.read_csv(kcore, names=["u", "v"], sep="\\s+")
        csk_nodes = pd.concat([csk["u"], csk["v"]]).drop_duplicates().to_numpy()

        # add the row to the dataframe
        generate_row(ikc_nodes, csk_nodes, node, edges, data, ikc_com_id)

    df = pd.DataFrame(data).fillna(0)
    df.columns = [
        "node",
        "jaccard",
        "ikc_com_id",
        "ikc_size",
        "ikc_interior",
        "ikc_boundary",
        "ikc_conductance",
        "ikc_density",
        "ikc_modularity",
        "ikc_connectivity",
        "ikc_mcd",
        "csk_size",
        "csk_interior",
        "csk_boundary",
        "csk_conductance",
        "csk_density",
        "csk_modularity",
        "csk_connectivity",
        "csk_mcd",
    ]
    df.to_csv(output, index=False)


if __name__ == "__main__":
    main()
