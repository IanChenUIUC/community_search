import itertools as it

import networkit as nk
import numpy as np
from common import (
    assert_permutationally_same,
    find_gt_comm,
    get_clique_chain,
)
from csk.algs.shell_baseline import AdvancedIndexBuilder


def test_shell_baseline_large():
    # cliques = np.array([3, 6, 5, 3, 5, 6, 12, 3, 7, 8, 7, 7, 8, 6, 4, 3, 4, 4, 8])
    cliques = np.array([3, 4, 3])
    num_nodes = np.sum(cliques)
    edges = get_clique_chain(cliques.tolist())
    vertices = np.arange(num_nodes)
    # cores = np.repeat(cliques, cliques) - 1

    row = edges.row.astype(int)
    col = edges.col.astype(int)
    data = edges.data.astype(int)
    graph = nk.GraphFromCoo((data, (row, col)))
    index = AdvancedIndexBuilder(graph)
    index.build()

    for v in vertices:
        query = np.array([v])
        _, et_comm = index.find_kcore(query, -1)
        gt_comm = find_gt_comm(cliques, query)
        assert_permutationally_same(et_comm, gt_comm)

    for vs in it.combinations(vertices, 2):
        query = np.array(list(vs))
        _, et_comm = index.find_kcore(query, -1)
        gt_comm = find_gt_comm(cliques, query)
        assert_permutationally_same(et_comm, gt_comm)
