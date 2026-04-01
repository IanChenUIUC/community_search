import numpy as np
from kcore_multi import run_queries, to_graph


def get_clique_chain(clique_sizes: list[int]):
    assert min(clique_sizes) >= 3

    edges = []
    offsets = np.cumsum([0] + clique_sizes[:-1])

    for i, size in enumerate(clique_sizes):
        start = offsets[i]
        # Intra-clique edges
        for u in range(start, start + size):
            for v in range(u + 1, start + size):
                edges.append([v, u])

        # Inter-clique bridge (last node of C_i to first node of C_i+1)
        if i < len(clique_sizes) - 1:
            u = start + size - 1
            v = offsets[i + 1]
            edges.append([u, v])

    edges = np.array(edges).T
    return edges


def generate_graph(cliques: list[int]):
    num_nodes = np.sum(cliques)
    edges = get_clique_chain(cliques)
    cores_array = np.repeat(cliques, cliques) - 1
    cores = dict(zip(np.arange(num_nodes), cores_array))
    return to_graph(edges, cores)


def assert_permutationally_same(a, b):
    assert a.shape == b.shape
    assert np.all(np.sort(a) == np.sort(b))


def test_graph_conversion():
    def remap_id(edges, cores, old, new):
        edges[edges == old] = new
        cores[new] = cores[old]
        del cores[old]

    cliques = np.array([3, 4])
    num_nodes = np.sum(cliques)
    edges = get_clique_chain(cliques.tolist())
    cores_array = np.repeat(cliques, cliques) - 1
    cores = dict(zip(np.arange(num_nodes), cores_array))

    # remapping some arbitrary node IDs
    remap_id(edges, cores, 2, 232)
    remap_id(edges, cores, 4, 136)

    graph, remapping = to_graph(edges, cores)
    formapping = dict(zip(remapping, np.arange(num_nodes)))

    # FIXME: write the test instead of just printing it out lol
    print()
    print(graph.todense())
    print(remapping)
    print(formapping)


def test_ksearch_singleton_simple():
    cliques = [3]
    num_nodes = np.sum(cliques)
    cores = np.repeat(cliques, cliques) - 1
    graph, _ = generate_graph(cliques)

    # work with the newIDs
    vertices = np.arange(num_nodes)

    # comms for 2-cores is the whole graph
    for q in vertices:
        comms = run_queries(graph, [np.array([q])], cores)
        _, comm = next(comms)
        assert_permutationally_same(comm, vertices)


def test_ksearch_singleton_a():
    cliques = [3, 4, 5, 4, 3]
    num_nodes = np.sum(cliques)
    graph, remapping = generate_graph(cliques)
    cores = np.repeat(cliques, cliques) - 1

    vertices = np.arange(num_nodes)
    formapping = dict(zip(remapping, np.arange(num_nodes)))
    remapped_cores = cores[remapping[vertices]]

    for q in range(num_nodes):
        gt_comm = vertices[cores >= cores[q]]

        query = np.array([formapping[q]])
        et_comms = run_queries(graph, [query], remapped_cores)
        _, et_comm = next(et_comms)
        et_comm = remapping[et_comm]

        # print(query, np.sort(et_comm), gt_comm)
        assert_permutationally_same(et_comm, gt_comm)


def test_ksearch_singleton_b():
    cliques = [5, 4, 3, 4, 5]
    num_nodes = np.sum(cliques)
    graph, remapping = generate_graph(cliques)
    cores = np.repeat(cliques, cliques) - 1

    vertices = np.arange(num_nodes)
    formapping = dict(zip(remapping, np.arange(num_nodes)))
    remapped_cores = cores[remapping[vertices]]

    gt_comms = []
    for _ in range(5):
        gt_comms.append(np.arange(5))
    for _ in range(4):
        gt_comms.append(np.arange(9))
    for _ in range(3):
        gt_comms.append(np.arange(21))
    for _ in range(4):
        gt_comms.append(np.arange(12, 21))
    for _ in range(4):
        gt_comms.append(np.arange(16, 21))

    for q, gt_comm in enumerate(gt_comms):
        query = np.array([formapping[q]])
        et_comms = run_queries(graph, [query], remapped_cores)
        _, et_comm = next(et_comms)
        et_comm = remapping[et_comm]

        # print(query, et_comm, gt_comm)
        assert_permutationally_same(et_comm, gt_comm)
