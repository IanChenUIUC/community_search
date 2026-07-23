import pathlib

import click
import polars as pl

from commsearch import Graph, ShellStruct


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("shell_base_path", type=click.Path(dir_okay=False))
@click.argument("coredecomp", type=click.Path(dir_okay=False, exists=True))
def main(indptr_path, indices_path, coredecomp, shell_base_path):
    graph = Graph.load(indptr_path, indices_path, warm=True)
    cores = pl.read_csv(
        coredecomp,
        has_header=False,
        new_columns=["node_id", "core"],
        schema_overrides={"core": pl.UInt64},
    ).sort("node_id")
    scores = cores.get_column("core").to_numpy()
    shell = ShellStruct.build(graph, scores)

    components_path = pathlib.Path(shell_base_path).with_suffix(".components.feather")
    tree_path = pathlib.Path(shell_base_path).with_suffix(".tree.feather")
    shell.save(components_path, tree_path)


if __name__ == "__main__":
    main()
