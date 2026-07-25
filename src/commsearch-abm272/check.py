"""
Checks the ABM network and the generated fields for agents.
"""

import click
import numpy as np
import polars as pl

from build_fields import scan_nodelist, field_code, LEGEND


def node_forest(nodelist_path):
    """Dense ``(n, year, gen, exists)`` over ``[0..maxid]``; ``gen == -1`` for seeds / no generator."""
    nl = (
        scan_nodelist(nodelist_path)
        .select(
            pl.col("node_id").cast(pl.Int64),
            pl.col("year").cast(pl.Int32),
            pl.col("generator_node_string").cast(pl.Int32, strict=False).fill_null(-1),
        )
        .collect()
    )
    n = nl["node_id"].max() + 1
    dense = pl.DataFrame({"node_id": pl.arange(0, n, eager=True)}).join(
        nl, on="node_id", how="left", maintain_order="left"
    )
    year = dense["year"].fill_null(-1).to_numpy()
    gen = dense["generator_node_string"].fill_null(-1).to_numpy()
    exists = dense["year"].is_not_null().to_numpy()
    return n, year, gen, exists


def seed_codes(tc_nodelist_path, n):
    """Dense int8 seed-code array (``-1`` off the seeds)."""
    tc = pl.read_csv(tc_nodelist_path, columns=["integer_id", "field"])
    code = np.full(n, -1, dtype=np.int8)
    code[tc["integer_id"].to_numpy()] = np.fromiter(
        (field_code(f) for f in tc["field"]), dtype=np.int8, count=tc.height
    )
    return code


def check_year_monotonic(nodelist_path):
    _, year, gen, _ = node_forest(nodelist_path)
    agents = np.nonzero(gen >= 0)[0]
    bad = agents[year[gen[agents]] >= year[agents]]
    assert bad.size == 0, f"{bad.size} agents not later than their generator"


def check_rooted_tree(nodelist_path, tc_nodelist_path):
    n, _, gen, exists = node_forest(nodelist_path)
    code = seed_codes(tc_nodelist_path, n)
    agents = np.nonzero(gen >= 0)[0]
    assert np.all(gen[agents] < agents), "a generator is not a lower id than its node"
    assert np.all(exists[gen[agents]]), "a generator points at a nonexistent (gap) node"
    orphans = np.nonzero(exists & (gen < 0) & (code < 0))[0]
    assert orphans.size == 0, f"{orphans.size} non seed nor agent nodes"


def check_output_correct(nodelist_path, tc_nodelist_path, all_nodelist_path):
    n, _, gen, exists = node_forest(nodelist_path)
    code = seed_codes(tc_nodelist_path, n)
    df = pl.read_csv(all_nodelist_path)
    assert df.height == n, f"output has {df.height} rows, expected {n}"
    assert np.array_equal(df["node_id"].to_numpy(), np.arange(n)), "node_id is not dense 0..maxid"

    field = df["field"].fill_null(-1).cast(pl.Int8).to_numpy()
    assert np.all(field[exists] >= 0) and np.all(field[~exists] < 0), "NULL field on seed or agent"
    seeds = np.nonzero(code >= 0)[0]
    assert np.array_equal(field[seeds], code[seeds]), "a seed's field does not match its tc code"
    agents = np.nonzero(gen >= 0)[0]
    assert np.array_equal(field[agents], field[gen[agents]]), "agent should clone generator field"


def log_summary(tc_nodelist_path, all_nodelist_path):
    name = {code: fields for fields, code in LEGEND.items()}
    field = pl.read_csv(all_nodelist_path, columns=["field"])["field"].fill_null(-1).to_numpy()

    codes, counts = np.unique(field[field >= 0], return_counts=True)
    click.echo("per-code histogram (5..14 = interdisciplinary descendants):")
    for c, ct in zip(codes.tolist(), counts.tolist()):
        click.echo(f"  {c:2d}  {name[c]:26s} {ct:>12d}")

    founders = pl.read_csv(tc_nodelist_path, columns=["integer_id", "role", "field"]).filter(
        pl.col("role") == "founder"
    )
    click.echo("founders (role == 'founder' in tc_combined):")
    for r in founders.iter_rows(named=True):
        got, expect = int(field[r["integer_id"]]), field_code(r["field"])
        flag = "OK" if got == expect else "MISMATCH"
        click.echo(f"  {r['integer_id']:6d}  {r['field']:14s} -> {got} ({name.get(got)}) [{flag}]")


@click.command()
@click.argument("nodelist_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("tc_nodelist_path", type=click.Path(exists=True, dir_okay=False))
@click.argument("all_nodelist_path", type=click.Path(exists=True, dir_okay=False))
def main(nodelist_path, tc_nodelist_path, all_nodelist_path):
    check_year_monotonic(nodelist_path)
    check_rooted_tree(nodelist_path, tc_nodelist_path)
    check_output_correct(nodelist_path, tc_nodelist_path, all_nodelist_path)

    log_summary(tc_nodelist_path, all_nodelist_path)


if __name__ == "__main__":
    main()
