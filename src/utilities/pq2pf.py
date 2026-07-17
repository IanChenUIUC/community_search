import click

import pyarrow as pa
import pyarrow.parquet as pq

### combine chunks will allocate a lot of memory, that can instead
### manually be written to a file-backed buffer for streaming.
### this is not done here for simplicity (just allocate more mem).


@click.command()
@click.option("-w", "--width", type=click.Choice([32, 64]), required=False, default=64)
@click.argument("files", nargs=-1, required=True)
def main(width, files):
    dtype = pa.uint64() if width == 64 else pa.uint32()
    for path in files:
        table = pq.read_table(path)
        table = table.combine_chunks()
        table = table.cast(pa.schema([(name, dtype) for name in table.column_names]))

        ### in https://github.com/apache/arrow/blob/main/python/pyarrow/includes/libarrow_feather.pxd#L32
        ### we see that chunksize is an `int`, so it overflows on tables `> 2 << 31` edges.
        ### thus, need to bypass the feather pyarrow.feather and directly use ipc, which
        ### correctly declares chunk_size as an int64_t.
        ### https://github.com/apache/arrow/blob/main/python/pyarrow/ipc.pxi#L610

        out = path.replace(".parquet", ".feather")
        # pf.write_feather(table, out, compression="uncompressed", chunksize=table.num_rows)

        opts = pa.ipc.IpcWriteOptions(allow_64bit=True)
        with pa.OSFile(out, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema, options=opts) as writer:
                writer.write_table(table)


if __name__ == "__main__":
    main()
