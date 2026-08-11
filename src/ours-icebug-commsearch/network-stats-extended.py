import click
import numpy as np
import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq

from analyze_shellstruct import subtree_sizes

import networkit as nk


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("components_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tree_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--fraction", default=0.01, type=float)
def main(indptr_path, components_path, tree_path, fraction):
    indptr = pq.read_table(indptr_path)["indptr"].combine_chunks().to_numpy()
    degrees = np.diff(indptr)

    tree_table = pf.read_table(tree_path, memory_map=True)
    components_table = pf.read_table(components_path, memory_map=True)

    coreness = tree_table.column("coreness").to_numpy()
    n, root = len(coreness), np.argmin(coreness)
    components = components_table.column(0).to_numpy()
    sizes = np.bincount(components, minlength=n)
    volumes = np.bincount(components, weights=degrees, minlength=n)

    indices = tree_table.column("csr_indices").combine_chunks().slice(0, n - 1)
    indptr = tree_table.column("csr_indptr").combine_chunks()
    indptr = pa.concat_arrays([indptr, pa.array([len(indices)], type=pa.uint64())])
    tree = nk.Graph.fromCSR(n, directed=False, out_indices=indices, out_indptr=indptr)

    cores_sizes = subtree_sizes(tree, root, sizes)
    cores_volumes = subtree_sizes(tree, root, volumes)

    # a query node's community is the connected k-core it lands in, i.e. the
    # subtree rooted at its shellstruct node
    k = max(1, round(fraction * len(degrees)))
    top = np.argpartition(degrees, -k)[-k:]
    roots = components[top]
    m = degrees.sum() / 2
    print(cores_sizes[roots].mean(), cores_volumes[roots].mean() / (2 * m))


if __name__ == "__main__":
    main()
