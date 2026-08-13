import itertools
import pathlib

import click
import pandas as pd

import common

COLUMNS = ["experiment", "network", "method", "stage", "size", "batch", "rep",
           "stat", "value", "status"]


NO_QUERY = object()


def emit(rows, key, mytime, status, query_s=NO_QUERY):
    """Append one long-format row per mytime stat, plus query_s for stages that have one."""
    for stat in common.MYTIME_KEYS:
        rows.append([*key, stat, (mytime or {}).get(stat), status])
    if query_s is not NO_QUERY:
        rows.append([*key, "query_s", query_s, status])


def shared_rows(network, out, states):
    """The per-network stages every online cell shares, with the cell axes left empty."""
    stages = [
        ("ib-core-decomp", "core-decomp",
         out / network / "testing-core-decomp" / "timing.txt",
         f"testing-core-decomp-{network}"),
        ("par-shellstruct", "offline",
         out / network / "testing-par-shellstruct" / "offline-timing.txt",
         f"testing-par-shellstruct-offline-{network}"),
        ("shellstruct", "offline",
         out / network / "testing-gullo-shellstruct" / "offline-timing.txt",
         f"testing-gullo-shellstruct-offline-{network}"),
    ]

    rows = []
    for method, stage, timing, node in stages:
        mytime = common.read_mytime(timing)
        emit(rows, ["testing", network, method, stage, None, None, None],
             mytime, common.row_status(node, mytime, states.get(node)))
    return rows


def steiner_rows(network, out, states, cells):
    """testing-steiner: one array task per cell, pycs querytimes."""
    d = out / network / "testing-steiner"
    rows = []
    for size, batch, rep in cells:
        cell = f"n{size}-b{batch}-rep{rep}"
        node = f"testing-steiner-{network}-{rep}-{size}-{batch}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        emit(rows, ["testing", network, "steiner", "online", size, batch, rep],
             mytime, common.row_status(f"{node} {cell}", mytime, states.get(node)),
             common.querytimes(d / f"querytimes-{cell}.csv", "pycs"))
    return rows


def par_shellstruct_rows(network, out, states, cells):
    """testing-par-shellstruct-online: one job walks every cell, so all cells share its state."""
    d = out / network / "testing-par-shellstruct"
    node = f"testing-par-shellstruct-online-{network}"
    task = states.get(node)

    rows = []
    for size, batch, rep in cells:
        cell = f"n{size}-b{batch}-rep{rep}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        emit(rows, ["testing", network, "par-shellstruct", "online", size, batch, rep],
             mytime, common.row_status(f"{node} {cell}", mytime, task),
             common.querytimes(d / f"querytimes-{cell}.csv", "pycs"))
    return rows


def gullo_rows(network, out, states, cells):
    """testing-gullo-shellstruct-online: timing comes from its own per-query log."""
    d = out / network / "testing-gullo-shellstruct"
    rows = []
    for size, batch, rep in cells:
        cell = f"n{size}-b{batch}-rep{rep}"
        node = f"testing-gullo-shellstruct-online-{network}-{rep}-{size}-{batch}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        emit(rows, ["testing", network, "shellstruct", "online", size, batch, rep],
             mytime, common.row_status(f"{node} {cell}", mytime, states.get(node)),
             common.gullo_timing(d / f"gullo-{cell}.log"))
    return rows


def csk_rows(network, out, states, cells, sizes):
    """testing-csk: declared for one query size only, and logs `queryid,ms` per query."""
    d = out / network / "testing-csk"
    rows = []
    for size, batch, rep in cells:
        if size not in sizes:
            continue
        cell = f"n{size}-b{batch}-rep{rep}"
        node = f"testing-csk-{network}-{rep}-{size}-{batch}"
        mytime = common.read_mytime(d / f"timing-{cell}.txt")
        emit(rows, ["testing", network, "csk", "online", size, batch, rep],
             mytime, common.row_status(f"{node} {cell}", mytime, states.get(node)),
             common.csk_timing(d / cell / "timing.log"))
    return rows


def testing_rows(spec, states, out):
    """Rows for the seven testing networks: shared stages per network, then online cells."""
    genquery = spec["recipe"]["genquery"]
    cells = list(itertools.product(genquery["sizes"], genquery["batches"],
                                   genquery["reps"]))
    csk_sizes = spec["recipe"]["testing-csk"]["params"]["size"]

    rows = []
    for network in spec["defaults"]["testing_networks"]:
        rows += shared_rows(network, out, states)
        rows += steiner_rows(network, out, states, cells)
        rows += par_shellstruct_rows(network, out, states, cells)
        rows += gullo_rows(network, out, states, cells)
        rows += csk_rows(network, out, states, cells, csk_sizes)
    return rows


def training_shared_rows(network, out, states):
    """The per-network stages training's online cells share."""
    stages = [
        ("ib-core-decomp", "core-decomp",
         out / network / "traincd-icebug-core-decomp" / "timing.txt",
         f"traincd-icebug-core-decomp-{network}"),
        ("par-shellstruct", "offline",
         out / network / "traincs" / "timing-shellstruct-offline.txt",
         f"traincs-steiner-shell-{network}"),
    ]

    rows = []
    for method, stage, timing, node in stages:
        mytime = common.read_mytime(timing)
        emit(rows, ["training", network, method, stage, None, None, None],
             mytime, common.row_status(node, mytime, states.get(node)))
    return rows


def training_batch_rows(network, out, states, sizes, reps):
    """traincs steiner and par-shellstruct run the 20 reps sequentially in one process per
    size, so their process stats belong to the size and only query_s varies by rep."""
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
            emit(rows, ["training", network, method, "online", size, 1, None],
                 mytime, status)
            for rep in reps:
                rows.append(["training", network, method, "online", size, 1, rep,
                             "query_s", (times or {}).get(rep), status])
    return rows


def training_local_rows(network, out, states, sizes, reps):
    """traincs-local: one array task per (rep, size, variant), each running a single query."""
    d = out / network / "traincs"
    rows = []
    for size, rep, variant in itertools.product(sizes, reps, ["local", "local-upper"]):
        node = f"traincs-local-{network}-{rep}-{size}-{variant}"
        mytime = common.read_mytime(d / f"timing-{variant}-n{size}-rep{rep}.txt")
        times = common.querytimes_rows(d / f"{variant}-querytimes-n{size}-rep{rep}.csv")
        emit(rows, ["training", network, variant, "online", size, 1, rep],
             mytime, common.row_status(node, mytime, states.get(node)),
             (times or {}).get(0))
    return rows


def training_rows(spec, states, out):
    """Rows for the training networks, whose three methods sit at three different grains."""
    genquery = spec["recipe"]["traincs-genquery"]
    reps, sizes = genquery["reps"], genquery["sizes"]

    rows = []
    for network in spec["defaults"]["training_networks"]:
        rows += training_shared_rows(network, out, states)
        rows += training_batch_rows(network, out, states, sizes, reps)
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

    genquery = spec["recipe"]["genquery"]
    common.report("testing", networks=len(spec["defaults"]["testing_networks"]),
                  reps=len(genquery["reps"]), sizes=len(genquery["sizes"]),
                  batches=len(genquery["batches"]))
    traincs = spec["recipe"]["traincs-genquery"]
    common.report("training", networks=len(spec["defaults"]["training_networks"]),
                  reps=len(traincs["reps"]), sizes=len(traincs["sizes"]))

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
