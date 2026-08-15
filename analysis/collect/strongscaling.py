import pathlib

import click
import pandas as pd

import common

COLUMNS = ["network", "method", "stage", "threads", "size", "batch", "rep",
           "stat", "value", "status"]

STAGE_AT = 2

DIR = "strongscaling"


def thread_shared(network, out, states, threads):
    """Prerequisite stages at one thread count, shared by both methods."""
    core = common.read_stage(out, states, network, DIR,
                             f"coredecomp-timing-t{threads}.txt",
                             f"strongscaling-core-decomp-{network}-{threads}")
    offline = common.read_stage(out, states, network, DIR,
                                f"shellstruct-timing-t{threads}.txt",
                                f"strongscaling-shellstruct-{network}-{threads}")
    return {"steiner": [("core-decomp", *core)],
            "par-shellstruct": [("core-decomp", *core), ("offline", *offline)]}


def steiner_rows(network, out, states, threads, reps, size, batch, shared):
    """strongscaling-steiner: one job per thread count, but its driver runs each rep as its
    own process, so every rep has its own mytime and querytimes file."""
    d = out / network / DIR
    node = f"strongscaling-steiner-{network}-{threads}"
    task = states.get(node)

    rows = []
    for rep in reps:
        mytime = common.read_mytime(d / f"steiner-timing-t{threads}-rep{rep}.txt")
        key = [network, "steiner", "online", threads, size, batch, rep]
        common.emit(rows, key, mytime,
                    common.row_status(f"{node} rep{rep}", mytime, task),
                    common.querytimes(d / f"steiner-querytimes-t{threads}-rep{rep}.csv",
                                      "pycs"))
        common.emit_shared(rows, key, shared, STAGE_AT)
    return rows


def par_shellstruct_rows(network, out, states, threads, reps, size, batch, shared):
    """query_shellstruct.py is serial, so strongscaling-shellstruct-online is declared
    without a thread axis: its one measurement repeats across the axis while the
    prerequisite stages it is paired with scale."""
    d = out / network / DIR
    node = f"strongscaling-shellstruct-online-{network}"
    task = states.get(node)

    rows = []
    for rep in reps:
        mytime = common.read_mytime(d / f"shellstruct-online-timing-rep{rep}.txt")
        key = [network, "par-shellstruct", "online", threads, size, batch, rep]
        common.emit(rows, key, mytime,
                    common.row_status(f"{node} rep{rep}", mytime, task),
                    common.querytimes(d / f"shellstruct-online-querytimes-rep{rep}.csv",
                                      "pycs"))
        common.emit_shared(rows, key, shared, STAGE_AT)
    return rows


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the thread sweep into analysis/strongscaling.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    out = root / "output"
    csv = root / "analysis" / "strongscaling.csv"

    genquery = spec["recipe"]["strongscaling-genquery"]
    size, batch = genquery["size"], genquery["batch"]
    reps = range(genquery["nreps"])

    rows = []
    for network in spec["defaults"]["all_networks"]:
        for threads in spec["defaults"]["threadcounts"]:
            shared = thread_shared(network, out, states, threads)
            rows += steiner_rows(network, out, states, threads, reps, size, batch,
                                 shared["steiner"])
            rows += par_shellstruct_rows(network, out, states, threads, reps, size, batch,
                                         shared["par-shellstruct"])

    df = pd.DataFrame(rows, columns=COLUMNS)
    df[["threads", "size", "batch", "rep"]] = \
        df[["threads", "size", "batch", "rep"]].astype("Int64")
    df.to_csv(csv, index=False)

    cells = df.drop_duplicates(["network", "method", "stage", "threads", "rep"])
    print(f"wrote {csv}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab([cells.stage, cells.method], cells.status, margins=True,
                      margins_name="total").to_string())


if __name__ == "__main__":
    main()
