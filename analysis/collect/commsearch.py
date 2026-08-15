import itertools
import pathlib

import click
import pandas as pd

import common

COLUMNS = ["experiment", "network", "method", "stage", "size", "batch", "rep",
           "stat", "value", "status"]


STAGE_AT = 3


def testing_shared(network, out, states):
    """Prerequisite stages per online method, from each recipe's deps: gullo reads the raw
    edge list so it needs no core decomposition."""
    core = common.read_stage(out, states, network, "testing-core-decomp", "timing.txt",
                             f"testing-core-decomp-{network}")
    par = common.read_stage(out, states, network, "testing-par-shellstruct",
                            "offline-timing.txt",
                            f"testing-par-shellstruct-offline-{network}")
    gullo = common.read_stage(out, states, network, "testing-gullo-shellstruct",
                              "offline-timing.txt",
                              f"testing-gullo-shellstruct-offline-{network}")
    return {"steiner": [("core-decomp", *core)],
            "csk": [("core-decomp", *core)],
            "par-shellstruct": [("core-decomp", *core), ("offline", *par)],
            "shellstruct": [("offline", *gullo)]}


def steiner_rows(network, out, states, cells, shared):
    """testing-steiner: one array task per cell, pycs querytimes."""
    d = out / network / "testing-steiner"
    rows = []
    for size, batch, rep in cells:
        cell = f"n{size}-b{batch}-rep{rep}"
        node = f"testing-steiner-{network}-{rep}-{size}-{batch}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        key = ["testing", network, "steiner", "online", size, batch, rep]
        common.emit(rows, key, mytime,
                    common.row_status(f"{node} {cell}", mytime, states.get(node)),
                    common.querytimes(d / f"querytimes-{cell}.csv", "pycs"))
        common.emit_shared(rows, key, shared, STAGE_AT)
    return rows


def par_shellstruct_rows(network, out, states, cells, shared):
    """testing-par-shellstruct-online: one job walks every cell, so all cells share its state."""
    d = out / network / "testing-par-shellstruct"
    node = f"testing-par-shellstruct-online-{network}"
    task = states.get(node)

    rows = []
    for size, batch, rep in cells:
        cell = f"n{size}-b{batch}-rep{rep}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        key = ["testing", network, "par-shellstruct", "online", size, batch, rep]
        common.emit(rows, key, mytime,
                    common.row_status(f"{node} {cell}", mytime, task),
                    common.querytimes(d / f"querytimes-{cell}.csv", "pycs"))
        common.emit_shared(rows, key, shared, STAGE_AT)
    return rows


def gullo_rows(network, out, states, cells, shared):
    """testing-gullo-shellstruct-online: timing comes from its own per-query log."""
    d = out / network / "testing-gullo-shellstruct"
    rows = []
    for size, batch, rep in cells:
        cell = f"n{size}-b{batch}-rep{rep}"
        node = f"testing-gullo-shellstruct-online-{network}-{rep}-{size}-{batch}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        key = ["testing", network, "shellstruct", "online", size, batch, rep]
        common.emit(rows, key, mytime,
                    common.row_status(f"{node} {cell}", mytime, states.get(node)),
                    common.gullo_timing(d / f"gullo-{cell}.log"))
        common.emit_shared(rows, key, shared, STAGE_AT)
    return rows


def csk_rows(network, out, states, cells, sizes, shared):
    """testing-csk: declared for one query size only, and logs `queryid,ms` per query."""
    d = out / network / "testing-csk"
    rows = []
    for size, batch, rep in cells:
        if size not in sizes:
            continue
        cell = f"n{size}-b{batch}-rep{rep}"
        node = f"testing-csk-{network}-{rep}-{size}-{batch}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        key = ["testing", network, "csk", "online", size, batch, rep]
        common.emit(rows, key, mytime,
                    common.row_status(f"{node} {cell}", mytime, states.get(node)),
                    common.csk_timing(d / cell / "timing.log"))
        common.emit_shared(rows, key, shared, STAGE_AT)
    return rows


def testing_rows(spec, states, out):
    """Rows for the seven testing networks: shared stages per network, then online cells."""
    genquery = spec["recipe"]["genquery"]
    cells = list(itertools.product(genquery["sizes"], genquery["batches"],
                                   genquery["reps"]))
    csk_sizes = spec["recipe"]["testing-csk"]["params"]["size"]

    rows = []
    for network in spec["defaults"]["testing_networks"]:
        shared = testing_shared(network, out, states)
        rows += steiner_rows(network, out, states, cells, shared["steiner"])
        rows += par_shellstruct_rows(network, out, states, cells, shared["par-shellstruct"])
        rows += gullo_rows(network, out, states, cells, shared["shellstruct"])
        rows += csk_rows(network, out, states, cells, csk_sizes, shared["csk"])
    return rows


def training_shared(network, out, states):
    """Prerequisite stages per online method. query_local.py takes no coreness, so local
    and local-upper require nothing."""
    core = common.read_stage(out, states, network, "traincd-icebug-core-decomp",
                             "timing.txt", f"traincd-icebug-core-decomp-{network}")
    offline = common.read_stage(out, states, network, "traincs",
                                "timing-shellstruct-offline.txt",
                                f"traincs-steiner-shell-{network}")
    return {"steiner": [("core-decomp", *core)],
            "par-shellstruct": [("core-decomp", *core), ("offline", *offline)],
            "local": [], "local-upper": []}


def training_batch_rows(network, out, states, sizes, reps, shared):
    """traincs steiner and par-shellstruct pack the reps into one process per size, so each
    rep cell repeats that process's stats and takes its own query_s."""
    d = out / network / "traincs"
    node = f"traincs-steiner-shell-{network}"

    rows = []
    for size in sizes:
        methods = [("steiner", f"timing-steiner-n{size}.txt",
                    f"steiner-querytimes-n{size}.csv"),
                   ("par-shellstruct", f"timing-shellstruct-online-n{size}.txt",
                    f"shellstruct-querytimes-n{size}.csv")]
        for method, timing, querytimes in methods:
            mytime = common.read_mytime(d / timing)
            times = common.querytimes_rows(d / querytimes)
            status = common.row_status(f"{node} {method} n{size}", mytime,
                                       states.get(node))
            for rep in reps:
                key = ["training", network, method, "online", size, 1, rep]
                common.emit(rows, key, mytime, status, (times or {}).get(rep))
                common.emit_shared(rows, key, shared[method], STAGE_AT)
    return rows


def training_local_rows(network, out, states, sizes, reps):
    """traincs-local: one array task per (rep, size, variant), each running a single query."""
    d = out / network / "traincs"
    rows = []
    for size, rep, variant in itertools.product(sizes, reps, ["local", "local-upper"]):
        node = f"traincs-local-{network}-{rep}-{size}-{variant}"
        mytime = common.read_mytime(d / f"timing-{variant}-n{size}-rep{rep}.txt")
        times = common.querytimes_rows(d / f"{variant}-querytimes-n{size}-rep{rep}.csv")
        common.emit(rows, ["training", network, variant, "online", size, 1, rep],
                    mytime, common.row_status(node, mytime, states.get(node)),
                    (times or {}).get(0))
    return rows


def training_rows(spec, states, out):
    """Rows for the training networks, whose three methods sit at three different grains."""
    genquery = spec["recipe"]["traincs-genquery"]
    reps, sizes = genquery["reps"], genquery["sizes"]

    rows = []
    for network in spec["defaults"]["training_networks"]:
        shared = training_shared(network, out, states)
        rows += training_batch_rows(network, out, states, sizes, reps, shared)
        rows += training_local_rows(network, out, states, sizes, reps)
    return rows


@click.command()
@click.option("--root", default=pathlib.Path(__file__).resolve().parents[2],
              type=click.Path(exists=True, file_okay=False),
              help="repo root holding slurm/, output/ and analysis/")
def main(root):
    """Collect the community search experiments into analysis/commsearch.csv."""
    root = pathlib.Path(root)
    spec = common.load_spec(root / "slurm" / "pipeline.toml")
    states = common.task_states(root / "slurm" / ".pipeline" / "run.jsonl")
    out = root / "output"
    csv = root / "analysis" / "commsearch.csv"

    df = pd.DataFrame(testing_rows(spec, states, out)
                      + training_rows(spec, states, out), columns=COLUMNS)
    df[["size", "batch", "rep"]] = df[["size", "batch", "rep"]].astype("Int64")
    df.to_csv(csv, index=False)

    cells = df.drop_duplicates(["experiment", "network", "method", "stage",
                                "size", "batch", "rep"])
    print(f"wrote {csv}: {len(df)} rows, {len(cells)} cells")
    print(pd.crosstab([cells.experiment, cells.stage, cells.method],
                      cells.status, margins=True, margins_name="total").to_string())


if __name__ == "__main__":
    main()
