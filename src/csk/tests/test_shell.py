import itertools as it

import numpy as np
from common import (
    assert_permutationally_same,
    find_gt_comm,
    get_clique_chain,
    to_triu_adj,
)
from csk.algs.shell import ShellStruct, search


def test_shell_singleton_a():
    cliques = np.array([3, 4, 5, 4, 3])
    num_nodes = np.sum(cliques)
    cores = np.repeat(cliques, cliques) - 1
    edges = get_clique_chain(cliques.tolist())

    graph, new_to_old, old_to_new = to_triu_adj(edges, cores)
    new_vs = np.arange(num_nodes, dtype=np.int32)
    new_ks = cores[new_to_old]
    shell = ShellStruct.build(graph, old_to_new, new_to_old, new_ks)
    shell.draw_tree()

    queries = [np.array([old_to_new[q]]) for q in new_vs]
    et_comms = list(search(shell, queries))
    assert len(et_comms) == len(queries)

    cached_comms: dict[int, np.ndarray] = {}
    for comm in et_comms:
        gt_comm = find_gt_comm(cliques, new_to_old[queries[comm.queryID]])
        if comm.commID in cached_comms:
            et_comm = cached_comms[comm.commID]
        else:
            et_comm = new_to_old[comm.vertices]
            cached_comms[comm.commID] = et_comm
        assert_permutationally_same(et_comm, gt_comm)


def test_shell_singleton_b():
    cliques = np.array([5, 4, 3, 4, 5])
    num_nodes = np.sum(cliques)
    cores = np.repeat(cliques, cliques) - 1
    edges = get_clique_chain(cliques.tolist())

    graph, new_to_old, old_to_new = to_triu_adj(edges, cores)
    new_vs = np.arange(num_nodes, dtype=np.int32)
    new_ks = cores[new_to_old]
    shell = ShellStruct.build(graph, old_to_new, new_to_old, new_ks)
    shell.draw_tree()

    queries = [np.array([old_to_new[q]]) for q in new_vs]
    et_comms = list(search(shell, queries))
    assert len(et_comms) == len(queries)

    cached_comms: dict[int, np.ndarray] = {}
    for comm in et_comms:
        gt_comm = find_gt_comm(cliques, new_to_old[queries[comm.queryID]])
        if comm.commID in cached_comms:
            et_comm = cached_comms[comm.commID]
        else:
            et_comm = new_to_old[comm.vertices]
            cached_comms[comm.commID] = et_comm
        assert_permutationally_same(et_comm, gt_comm)


def test_shell_pairs():
    cliques = np.array([3, 6, 5, 3, 5, 6, 12, 3, 7, 8, 7, 7, 8, 6, 4, 3, 4, 4, 8])
    num_nodes = np.sum(cliques)
    cores = np.repeat(cliques, cliques) - 1
    edges = get_clique_chain(cliques.tolist())

    graph, new_to_old, old_to_new = to_triu_adj(edges, cores)
    new_vs = np.arange(num_nodes, dtype=np.int32)
    new_ks = cores[new_to_old]
    shell = ShellStruct.build(graph, old_to_new, new_to_old, new_ks)
    shell.draw_tree()

    pairs = it.combinations(new_vs, 2)
    queries = [np.array(old_to_new[[q1, q2]]) for q1, q2 in pairs]
    et_comms = list(search(shell, queries))
    assert len(et_comms) == len(queries)

    cached_comms: dict[int, np.ndarray] = {}
    for comm in et_comms:
        gt_comm = find_gt_comm(cliques, new_to_old[queries[comm.queryID]])
        if comm.commID in cached_comms:
            et_comm = cached_comms[comm.commID]
        else:
            et_comm = new_to_old[comm.vertices]
            cached_comms[comm.commID] = et_comm
        assert_permutationally_same(et_comm, gt_comm)
