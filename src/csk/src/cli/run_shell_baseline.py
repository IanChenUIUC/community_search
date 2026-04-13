import pickle
import time
from pathlib import Path

import click
import networkit as nk

from ..csk.shell_baseline import AdvancedIndexBuilder


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--output", required=True, type=click.Path())
def shell_baseline_index(edgelist, output):
    reader = nk.graphio.EdgeListReader("\t", 0, continuous=False)
    graph = reader.read(edgelist)
    # getNodeMap: original_id (str) -> internal_id (int)
    node_map = reader.getNodeMap()
    # Build reverse map: internal_id -> original_id (as int)
    reverse_map = {v: int(k) for k, v in node_map.items()}
    print(f"Graph loaded: {graph.numberOfNodes()} nodes, {graph.numberOfEdges()} edges")

    # obtain index
    start = time.perf_counter()
    indexer = AdvancedIndexBuilder(graph)
    indexer.build()
    end = time.perf_counter()
    print(end - start)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as f:
        pickle.dump(
            {"indexer": indexer, "reverse_map": reverse_map, "node_map": node_map},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


@click.command()
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--nodelist", required=True, type=click.Path(exists=True))
@click.option("--outputdir", required=True, type=click.Path())
def shell_baseline_search(index, nodelist, outputdir):
    with open(index, "rb") as f:
        data = pickle.load(f)
    # Support both old format (bare indexer) and new format (dict with maps)
    if isinstance(data, dict):
        indexer = data["indexer"]
        reverse_map = data.get("reverse_map")  # internal_id -> original_id
        node_map = data.get("node_map")  # original_id (str) -> internal_id
    else:
        indexer = data
        reverse_map = None
        node_map = None
    print("Indexer loaded")

    indexer.draw()

    count = 0
    with open(nodelist) as nodefile:
        for line in nodefile.readlines():
            parts = line.strip().split(" ")
            queries_str = parts[0]
            k = int(parts[1]) if len(parts) > 1 else -1
            original_queries = [int(q) for q in queries_str.split(",")]
            # Map original IDs to internal IDs if node_map exists
            if node_map is not None:
                queries = []
                for q in original_queries:
                    if str(q) in node_map:
                        queries.append(node_map[str(q)])
                    else:
                        print(f"Warning: query node {q} not found in graph")
                        queries.append(-1)  # will be caught by find_kcore
            else:
                queries = original_queries

            resolved_k, component = indexer.find_kcore(queries, k)
            # Map internal IDs back to original IDs in output
            if reverse_map is not None:
                component = [reverse_map.get(v, v) for v in component]
            # Use query string as dir name, but fall back to a descriptive
            # name derived from the nodelist filename when it's too long for
            # the filesystem (max 255 bytes).
            dirname = queries_str
            if len(dirname.encode("utf-8")) > 255:
                dirname = Path(nodelist).stem + f"_line{count}"
            outpath = Path(outputdir) / f"{dirname}/kcore_k{resolved_k}.txt"
            outpath.parent.mkdir(parents=True, exist_ok=True)

            with outpath.open("w") as outfile:
                outfile.write("\n".join(map(str, component)))
                if len(component):
                    outfile.write("\n")
                # outfile.write("-1")

            count += 1
            if count % 50 == 0:
                print(f"Finished {count} queries...")

    print("Search jobs completed.")
