import pyarrow.feather as pf
import pyarrow.parquet as pq
import sys

for path in sys.argv[1:]:
    table = pq.read_table(path)
    table = table.combine_chunks()
    out = path.replace(".parquet", ".feather")
    pf.write_feather(table, out, compression="uncompressed", chunksize=table.num_rows)
