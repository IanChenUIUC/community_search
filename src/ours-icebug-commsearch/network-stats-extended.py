import csv
import pathlib

import click
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import pyarrow.parquet as pq

import networkit as nk


def subtree_sizes(tree, root, sizes):
    retval = sizes.copy()

    order, stack = [], [root]
    seen = set()
    while stack:
        u = stack.pop()
        order.append(u)
        for w in tree.iterNeighbors(u):
            if w in seen:
                continue
            seen.add(w)
            stack.append(w)

    for u in reversed(order):
        for w in tree.iterNeighbors(u):
            retval[u] += retval[w]

    return retval


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("components_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tree_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_csv", type=click.Path(dir_okay=False))
@click.argument("coresizes_csv", type=click.Path(dir_okay=False))
@click.option("--fraction", default=0.01, type=float)
def main(indptr_path, components_path, tree_path, output_csv, coresizes_csv, fraction):
    indptr = pq.read_table(indptr_path)["indptr"].combine_chunks().to_numpy()
    degrees = np.diff(indptr)

    tree_table = pf.read_table(tree_path, memory_map=True)
    components_table = pf.read_table(components_path, memory_map=True)

    coreness = tree_table.column("coreness").to_numpy()
    n, root = len(coreness), np.argmin(coreness)
    components = components_table.column(0).to_numpy()
    sizes = np.bincount(components, minlength=n)
    volumes = np.bincount(components, weights=degrees, minlength=n)

    children = tree_table.column("children").combine_chunks()
    tree = nk.Graph.fromCSR(
        n, directed=False, out_indices=children.values, out_indptr=children.offsets
    )

    cores_sizes = subtree_sizes(tree, root, sizes)
    cores_volumes = subtree_sizes(tree, root, volumes)

    # a query node's community is the connected k-core it lands in, i.e. the
    # subtree rooted at its shellstruct node
    k = max(1, round(fraction * len(degrees)))
    top = np.argpartition(degrees, -k)[-k:]
    roots = components[top]
    m = degrees.sum() / 2

    stats = {
        "communities": int((coreness != 0).sum()),
        "avg_community_size": float(cores_sizes[roots].mean()),
        "avg_community_size_fraction": float(cores_sizes[roots].mean() / len(degrees)),
        "avg_volume_fraction": float(cores_volumes[roots].mean() / (2 * m)),
    }
    pathlib.Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stat", "value"])
        writer.writerows(stats.items())
    for stat, value in stats.items():
        print(f"{stat} = {value}")

    mask = coreness != 0
    pathlib.Path(coresizes_csv).parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(dict(core=coreness[mask], sizes=cores_sizes[mask])).to_csv(
        coresizes_csv, index=False, header=True)


if __name__ == "__main__":
    main()
