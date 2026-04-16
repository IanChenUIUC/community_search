import networkit as nk

"""
K-core with an index structure based on ShellStruct (N. Barbieri, F. Bonchi, E. Galimberti, and F. Gullo. Efficient and effective community search. DMKD, 29(5):1406–1433, 2015.)
Implement the advanced algorithm to build the index (Y. Fang, R. Cheng, S. Luo, and J. Hu. Effective community search for large attributed graphs. PVLDB, 9(12):1233–1244, Aug. 2016.)
"""


# Disjoint set which stores anchor nodes
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}  # each node starts as its own parent

    def find(self, item):
        if self.parent[item] == item:
            return item
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, set1, set2):
        root1 = self.find(set1)
        root2 = self.find(set2)
        # merge if the two roots are different. TODO: optimize for balanced tree
        if root1 != root2:
            self.parent[root1] = root2


class CLTreeNode:
    def __init__(self, core_num, vertices):
        self.core_num = core_num
        self.vertex_set = set(vertices)
        self.children = set()
        self.parent = None

    def add_child(self, child):
        self.children.add(child)
        child.parent = self


class AdvancedIndexBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.v_to_treenode = dict()  # node_id -> CL-tree node
        self.anchor_map = (
            dict()
        )  # root -> anchor node_id (i.e., the node with least core number in the set)

    def __getstate__(self):
        """Iteratively flatten the CL-tree for pickle serialization,
        avoiding deep recursion that causes SIGSEGV on large graphs."""
        # Assign each CLTreeNode a unique integer id via BFS
        node_to_id = {}
        # Collect all tree nodes reachable from v_to_treenode values
        for treenode in self.v_to_treenode.values():
            if id(treenode) not in node_to_id:
                # Walk up to the root first
                root = treenode
                while root.parent is not None:
                    root = root.parent
                # BFS from root
                bfs = [root]
                while bfs:
                    cur = bfs.pop(0)
                    if id(cur) not in node_to_id:
                        node_to_id[id(cur)] = len(node_to_id)
                        bfs.extend(cur.children)

        {v: k for k, v in node_to_id.items()}
        # Build a flat list: each entry is (core_num, vertex_set, parent_id, [child_ids])
        obj_map = {}  # python id(obj) -> obj, for lookup
        for treenode in self.v_to_treenode.values():
            obj_map[id(treenode)] = treenode
            if treenode.parent is not None:
                obj_map[id(treenode.parent)] = treenode.parent
        # BFS again to capture all nodes including those not in v_to_treenode
        for py_id in list(node_to_id.keys()):
            if py_id in obj_map:
                for child in obj_map[py_id].children:
                    obj_map[id(child)] = child

        flat_nodes = [None] * len(node_to_id)
        for py_id, idx in node_to_id.items():
            node = obj_map[py_id]
            parent_idx = node_to_id[id(node.parent)] if node.parent is not None else -1
            child_idxs = [node_to_id[id(c)] for c in node.children]
            flat_nodes[idx] = (node.core_num, node.vertex_set, parent_idx, child_idxs)

        # Map v_to_treenode: vertex -> flat index
        v_to_idx = {v: node_to_id[id(tn)] for v, tn in self.v_to_treenode.items()}

        return {"flat_nodes": flat_nodes, "v_to_idx": v_to_idx}

    def __setstate__(self, state):
        """Reconstruct the CL-tree iteratively from the flat representation."""
        flat_nodes = state["flat_nodes"]
        v_to_idx = state["v_to_idx"]

        # Rebuild CLTreeNode objects without parent/child links first
        tree_nodes = []
        for core_num, vertex_set, _, _ in flat_nodes:
            tree_nodes.append(CLTreeNode(core_num, vertex_set))

        # Wire up parent/child pointers
        for idx, (_, _, parent_idx, child_idxs) in enumerate(flat_nodes):
            node = tree_nodes[idx]
            if parent_idx != -1:
                node.parent = tree_nodes[parent_idx]
            for ci in child_idxs:
                node.children.add(tree_nodes[ci])

        # Rebuild v_to_treenode
        self.v_to_treenode = {v: tree_nodes[idx] for v, idx in v_to_idx.items()}
        self.graph = None
        self.anchor_map = {}

    def build(self):
        core = nk.centrality.CoreDecomposition(self.graph).run()
        k_max = core.maxCoreNumber()
        k_shells = core.getPartition()

        top_level_treenodes = set()

        # initialize union-find and anchor map
        nodelist = set()
        for v in self.graph.iterNodes():
            self.anchor_map[v] = v
            nodelist.add(v)
        uf = UnionFind(nodelist)
        count = 0

        # bottom-up: decrement k in each iteration
        for k in range(k_max, -1, -1):
            # (new) nodes encountered for this k
            # set of nodes with core number exactly being k
            k_shell_nodes = k_shells.getMembers(k)

            if not k_shell_nodes:
                continue  # no nodes for this layer

            # find connected components at this level, taking previous CCs into consideration
            k_prime_nodes = set()  # all connected components in k'-cores where k' < k
            for v in k_shell_nodes:
                k_prime_nodes.add(uf.find(v))

            # nodes used to compute k-core
            k_union_nodes = k_shell_nodes | k_prime_nodes
            k_union_subgraph = nk.graphtools.subgraphFromNodes(
                self.graph, k_union_nodes
            )
            k_union_cc = (
                nk.components.ConnectedComponents(k_union_subgraph)
                .run()
                .getComponents()
            )

            for cc_nodes in k_union_cc:
                cc_nodes = set(cc_nodes)
                cc_union_nodes = cc_nodes & k_shell_nodes

                if len(cc_union_nodes) == 0:
                    continue  # empty connected components

                # CL-tree node for this k-shell component
                k_cc_treenode = CLTreeNode(k, cc_union_nodes)
                top_level_treenodes.add(k_cc_treenode)
                count += 1

                # to prevent repetitively adding the same CL-tree node
                processed_treenodes = set()

                for v in cc_union_nodes:
                    self.v_to_treenode[v] = k_cc_treenode

                    for u in self.graph.iterNeighbors(v):
                        # update CL-tree
                        if core.score(u) > core.score(v):
                            u_root = uf.find(u)
                            u_anchor = self.anchor_map[u_root]
                            prev_treenode = self.v_to_treenode[u_anchor]

                            if (
                                prev_treenode in processed_treenodes
                                or prev_treenode is k_cc_treenode
                            ):
                                continue  # skip repetitive and identical treenode

                            k_cc_treenode.add_child(prev_treenode)
                            top_level_treenodes.remove(prev_treenode)
                            processed_treenodes.add(prev_treenode)

                        # update union-find structure
                        if core.score(u) >= core.score(v):
                            uf.union(u, v)

                    # update anchor if v has a strictly smaller core number
                    v_root = uf.find(v)
                    if core.score(self.anchor_map[v_root]) > core.score(v):
                        self.anchor_map[v_root] = v

        root = CLTreeNode(-1, [])
        for treenode in top_level_treenodes:
            root.add_child(treenode)

        return None

    def get_root(self):
        node = self.v_to_treenode[0]
        while node.parent is not None:
            node = node.parent
        return node

    def draw(self, root=None):
        """
        Print an ASCII representation of the subtree rooted at `root`.
        Handles cycles by tracking visited nodes.

        NOTE: this is LLM generated code
        """

        if root is None:
            root = self.get_root()

        def _node_label(node):
            verts = ",".join(str(v) for v in sorted(node.vertex_set))
            return f"CLTreeNode(core={node.core_num}, vertices={{ {verts} }})"

        def _sorted_children(node):
            # deterministic ordering: higher core first, then smallest vertex id
            def key(c):
                minv = min(c.vertex_set) if c.vertex_set else float("inf")
                return (-c.core_num, minv)

            return sorted(node.children, key=key)

        visited = set()

        def _draw(node, prefix="", is_last=True):
            nid = id(node)
            connector = "└─ " if is_last else "├─ "
            line = prefix + connector + _node_label(node)
            print(line)
            if nid in visited:
                # indicate already-printed node to avoid infinite loops
                print(prefix + ("    " if is_last else "│   ") + "[already shown]")
                return
            visited.add(nid)

            children = _sorted_children(node)
            for i, child in enumerate(children):
                last = i == len(children) - 1
                new_prefix = prefix + ("    " if is_last else "│   ")
                _draw(child, new_prefix, last)

        # root printed without leading connector
        print(_node_label(root))
        visited.add(id(root))
        children = _sorted_children(root)
        for i, child in enumerate(children):
            _draw(child, "", i == len(children) - 1)

    def ancestors(self, node):
        """Return the set of all ancestors (including the node itself)."""
        result = set()
        while node is not None:
            result.add(node)
            node = node.parent
        # print(f"Ancestor found for node {node}")
        return result

    def find_kcore(self, queries, k):
        """Returns (resolved_k, sorted_vertices). resolved_k is the actual
        core number used (matters when k=-1)."""
        nodes = []
        for q in queries:
            if q not in self.v_to_treenode:
                return k, []
            node = self.v_to_treenode[q]
            if k == -1:
                # just stay at the node - this is the highest k possible
                pass
            elif node.core_num < k:
                return k, []
            else:
                # walk up to the first ancestor with core_num <= k
                while node.parent and node.parent.core_num >= k:
                    node = node.parent
            nodes.append(node)

        # intersecting ancestor sets to find LCA
        lca_candidates = self.ancestors(nodes[0])
        for node in nodes[1:]:
            lca_candidates &= self.ancestors(node)

        if not lca_candidates:
            return k, []

        # LCA is the deepest common ancestor (highest core_num)
        lca = max(lca_candidates, key=lambda n: n.core_num)

        if k != -1 and lca.core_num < k:
            return k, []

        # collect all vertices from the LCA and its descendants
        vertices = set()
        stack = [lca]
        while stack:
            cur = stack.pop()
            vertices.update(cur.vertex_set)
            stack.extend(cur.children)
        return lca.core_num, sorted(vertices)
