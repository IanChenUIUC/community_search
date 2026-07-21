# K-Core Community Search

Input: a graph and a list of queries; each query being a connected set of vertices in G
Output: the k-core community for each query (i.e. a list of array_like objects).
  The coreness of a query is the largest k s.t. the k-core of the graph has all of Q and Q is connected in that k-core.
  The k-core community is the *maximal* subgraph that has this coreness (i.e. it is a connected component of some k-core of the graph).

Both algorithms here have an offline phase and an online phase.
SteinerkCore offline is external (a standard core decomposition), and ShellStruct builds it here.

The graph structure will always have indices from 0 to n-1.
We will take in as CSR format, a .feather or .parquet indices/indptr.
The adjacency list will be sorted.

## SteinerKCore search

For a single query vertex, we can do a BFS to find the component (using the coreness as the threshold).
For a single query set, we add all into a *priority queue*, with priorities as the coreness of a vertex.
Then, we expand outwards until we make it connected (via a union-find data structure).
Finally, make it maximal by finishing the round only when the largest key of the PQ drops (or the PQ is empty).

Finally, for a multiple queries, we use a counts datastructure (for each component/query combination, we main how many query vertices are in that component).
This gives us a logarithmic cost to maintaining these sequentially.

All algorithms are sequential (but numpy operations may be vectorized).

## ShellStruct search

If we look at the previous structure in SteinerKCore, we are essentially building a bottleneck steiner tree.
We can actually store this as a rooted tree in which going down has increasing oreness, and each node in that tree represents a component.
The online phase is just to use that sparse table (LCA --> euler tour --> RMQ reduction) to do O(1) queries, and then find the vertices by a traversal downwards.

The algorithm is as follows:

1. iterate through the k-shells in decreasing k
2. in parallel, find the set of roots of all v that are adjacent to u in the k-shell, and coreness(v) > k
    these are exactly the ones that we need to make children for.
3. in parallel, union all u with its adjacency
4. in parallel, add u into a new TreeNode associated with uf.find(u)
5. make a tree-node with k=0 (if necessary)
6. build the LCA
