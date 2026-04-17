import click
import polars as pl


@click.command()
@click.option("--edgelist", required=True, type=click.Path(exists=True))
@click.option("--out_edges", required=True, type=click.Path())
@click.option("--out_mapping", required=True, type=click.Path())
@click.option("--sep", required=False, type=str, default=",")
@click.option("--head", required=False, type=bool, default=True)
def compact(edgelist, out_edges, out_mapping, sep, head):
    cols = ["u", "v"]
    e = pl.scan_csv(edgelist, has_header=head, separator=sep, new_columns=cols)

    mapping = (
        pl.concat([e.select(pl.col("u").alias("w")), e.select(pl.col("v").alias("w"))])
        .unique()
        .with_row_index("new_id")
        .collect(engine="streaming")
    )
    mapping.select("w").write_csv(out_mapping, include_header=False)

    mapping_lz = mapping.lazy()
    (
        e.join(mapping_lz, left_on="u", right_on="w", how="inner")
        .rename({"new_id": "new_u"})
        .join(mapping_lz, left_on="v", right_on="w", how="inner")
        .rename({"new_id": "new_v"})
        .select(["new_u", "new_v"])
        .sink_csv(out_edges, include_header=False)
    )
