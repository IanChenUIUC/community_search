# About

These tools were obtained from Prof. Fang's website, which is not available / seen anymore.
These are the codes that are used in [vidya-ms-thesis], slightly-adapted to handle batched single-vertex queries and more flexibility in the file I/O.

## Usage

```
$ ./main
Usage: ./main [COMMAND] [ARGS]...

./main -i maindir edgelist
Build k-core Index:
	@maindir is the directory where the output @maindir/kcore_index.txt will be created. maindir should end with a '/'.
	kcore_index is a two column file (csv, no header) with node and its core number

@edgelist is two column (csv, with header) unweighted graph. The node ids are from 0 to N-1 (memory will be allocated for up to vertex_id_max nodes).
	./main -s maindir edgelist indexfile nodes --nodelist
Search k-core Communities:
	@maindir [see comments above]
	@edgelist [see comments above]
	@indexfile is the output of the -i command above.
	@nodes is a two column file of node and minimum core number.
	--nodelist is an option that outputs the nodes (intead of edges) of the community.
```
