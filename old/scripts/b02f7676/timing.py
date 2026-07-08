from contextlib import suppress
from datetime import datetime
from pathlib import Path

import pandas as pd


def parse_hms(time_str):
    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S"):
        with suppress(ValueError):
            value = datetime.strptime(time_str, fmt).time()
            return value

    raise ValueError(f"Unrecognized time format: {time_str}")


def get_wall(filepath):
    with open(filepath) as f:
        lines = [line.strip() for line in f.readlines() if "wall clock" in line]

    if len(lines) != 1:
        raise Exception(f"Could not find walltime in {filepath}")

    time_str = lines[0].strip().split(" ")[-1]

    try:
        t = parse_hms(time_str)
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6
    except Exception as e:
        raise Exception(f"Could not parse walltime {time_str} in {filepath}") from e


def main():
    basedir = "/u/ianchen3/ianchen3/csearch/output"

    index = [
        f"{basedir}/b02f7676/cpp/{{network}}/index.err",  # CSK (cpp, edgelist)
        f"{basedir}/b02f7676/cpp/{{network}}/index.err",  # CSK (cpp, nodelist)
        f"{basedir}/b02f7676/py/{{network}}/index.err",  # CSK (python)
    ]

    query = [
        f"{basedir}/b02f7676/cpp/{{network}}/edge/query.err",  # CSK (cpp, edgelist)
        f"{basedir}/b02f7676/cpp/{{network}}/node/query.err",  # CSK (cpp, nodelist)
        f"{basedir}/b02f7676/py/{{network}}/query.err",  # CSK (python)
    ]

    data = []

    with Path("./scripts/b02f7676/networks.txt").open() as networks:
        for network in networks.readlines():
            network = network.strip()

            try:
                for op, paths in zip(["index", "query"], [index, query]):
                    for impl, path in zip(["cpp_edge", "cpp_node", "python"], paths):
                        filepath = path.format(network=network)
                        data.append([network, op, impl, get_wall(filepath)])

                print(f"Processed in {network}", flush=True)
            except Exception as e:
                print(f"Error in {network}: {e}", flush=True)

    data = pd.DataFrame(data, columns=["network", "op", "impl", "time"])
    outfile = Path(f"{basedir}/b02f7676/analysis/timing.txt")
    outfile.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(outfile, index=None)


if __name__ == "__main__":
    main()
