import pyarrow as pa
import pyarrow.feather as pf
import pyarrow.parquet as pq
import sys

### combine chunks will allocate a lot of memory, that can instead
### manually be written to a file-backed buffer for streaming.
### this is not done here for simplicity (just allocate more mem).

for path in sys.argv[1:]:
    table = pq.read_table(path)
    table = table.combine_chunks()
    out = path.replace(".parquet", ".feather")

    ### in https://github.com/apache/arrow/blob/main/python/pyarrow/includes/libarrow_feather.pxd#L32
    ### we see that chunksize is an `int`, so it overflows on tables `> 2 << 31` edges.
    ### thus, need to bypass the feather pyarrow.feather and directly use ipc, which
    ### correctly declares chunk_size as an int64_t.
    ### https://github.com/apache/arrow/blob/main/python/pyarrow/ipc.pxi#L610

    # pf.write_feather(table, out, compression="uncompressed", chunksize=table.num_rows)

    with pa.OSFile(out, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
