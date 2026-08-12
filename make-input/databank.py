import argparse
import os
import subprocess
import sys
import tempfile

import format_conversion.format as fmt

OUTPUT_DIR = "/u/ianchen3/community_search/input"
BUILD_NODELIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../src/utilities/build_nodelist.py"
)

DATASETS = {
    "cen": {
        "url": "https://databank.illinois.edu/datafiles/52fdo/download",
        "spec": fmt.CsvEdgelist.Read(sep="\t", skip_rows=0),
        "nodelist_args": ["--format", "csv", "--sep", "\t", "--no-header"],
    },
    "abm14": {
        "url": "https://aws-databank-alb.library.illinois.edu/datafiles/lo4nh/download",
        "spec": fmt.EdgelistParquet.Read(),
        "nodelist_args": ["--format", "parquet"],
    },
}


def process(name, tmpdir):
    dataset = DATASETS[name]
    edges = os.path.join(tmpdir, f"{name}.edges")
    nodes = os.path.join(tmpdir, f"{name}.nodes.csv")

    print(f"[{name}] downloading {dataset['url']}", flush=True)
    subprocess.run(["wget", "-q", dataset["url"], "-O", edges], check=True)

    print(f"[{name}] building the node list", flush=True)
    subprocess.run(
        [sys.executable, BUILD_NODELIST, edges, nodes, *dataset["nodelist_args"]],
        check=True,
    )

    print(f"[{name}] converting -> {OUTPUT_DIR}/{name}.csv", flush=True)
    fmt.convert(
        fmt.GraphDescriptor(edges, dataset["spec"]),
        fmt.GraphDescriptor(os.path.join(OUTPUT_DIR, name), fmt.CsvEdgelist.Write()),
        nodes=fmt.NodeDescriptor(nodes, fmt.Nodelist.Csv(skip_rows=1)),
        sort_neighbors=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Write the Illinois Databank networks to ../input/<dataset>.csv."
    )
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"any of {', '.join(DATASETS)} (default: all of them)")
    args = parser.parse_args()

    unknown = sorted(set(args.datasets) - set(DATASETS))
    if unknown:
        parser.error(f"unknown dataset(s) {', '.join(unknown)}; "
                     f"choose from {', '.join(DATASETS)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        for name in args.datasets or DATASETS:
            process(name, tmpdir)


if __name__ == "__main__":
    main()
