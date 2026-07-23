import time

import click
import numpy as np
import polars as pl

from commsearch import Graph, ShellStruct, SteinerKCore


@click.command()
@click.argument("indptr_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("indices_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("coredecomp", type=click.Path(dir_okay=False, exists=True))
@click.argument("queries_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("-t", "--num_threads", type=int, default=1)
@click.option("-b", "--max_batch_size", type=int, default=1)
def main(indptr_path, indices_path, coredecomp, queries_path, output, num_threads, max_batch_size):
    graph = Graph.load(indptr_path, indices_path, warm=True)
    cores = pl.read_csv(coredecomp, has_header=False, new_columns=["id", "core"]).sort("id")
    scores = cores.get_column("core").to_numpy()
    steiner = SteinerKCore(graph, scores)
    steiner.warmup()

    with open(queries_path) as f:
        queries = [np.fromstring(line, sep=",") for line in f.readlines()]

    start = time.perf_counter()
    steiner.run_parallel(queries, num_threads, max_batch_size)
    end = time.perf_counter()

    timing = steiner.timing
    index = [str(x) for x in range(len(timing))]
    timing.append(end - start)
    index.append("all")

    df = pl.DataFrame({"index": index, "wall_s": timing})
    df.write_csv(output)


if __name__ == "__main__":
    main()
