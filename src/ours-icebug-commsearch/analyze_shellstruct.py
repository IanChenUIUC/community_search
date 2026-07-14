import pathlib

import click
import networkit as nk
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq


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
@click.argument("components_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tree_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
def main(components_path, tree_path, output_path):
    tree_table = pf.read_table(tree_path)
    components_table = pf.read_table(components_path)

    coreness = tree_table.column("coreness").to_numpy()
    n, root = len(coreness), np.argmin(coreness)
    sizes = np.bincount(components_table.column(0).to_numpy(), minlength=n)

    indices = tree_table.column("csr_indices").combine_chunks().slice(0, n - 1)
    indptr = tree_table.column("csr_indptr").combine_chunks()
    indptr = pa.concat_arrays([indptr, pa.array([len(indices)], type=pa.uint64())])
    tree = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    # each node in the shellstruct represents a distinct connected k-core
    # thus, summing the size of the subtree gets the sizes of each community
    cores_sizes = subtree_sizes(tree, root, sizes)

    pathlib.Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    mask = coreness != 0
    df = pd.DataFrame(dict(core=coreness[mask], sizes=cores_sizes[mask]))
    df.to_csv(output_path, index=False, header=True)


if __name__ == "__main__":
    main()
