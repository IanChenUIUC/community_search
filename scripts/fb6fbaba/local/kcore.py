import itertools as it
from pathlib import Path

import networkx as nx


def get_comm(g: nx.Graph, q: int, query_k: int) -> set[int]:
    def update_lowers(subg: nx.Graph, lowers, finished):
        # subg = nx.induced_subgraph(subg, set(subg.nodes).difference(frntr))
        degrees = dict(subg.degree)

        max_degree = max(degrees.values())
        queues = [set() for _ in range(max_degree + 1)]
        for v, degree in subg.degree:
            queues[degree].add(v)

        update = False
        for k in range(max_degree + 1):
            while queues[k]:
                v = queues[k].pop()
                if lowers[v] != k:
                    lowers[v] = k

                    update = v not in finished
                    if k >= query_k:
                        finished.add(v)

                for u in subg.neighbors(v):
                    if degrees[u] > k:
                        queues[degrees[u]].remove(u)
                        queues[degrees[u] - 1].add(u)
                        degrees[u] -= 1
        return update

    def update_uppers(subg: nx.Graph, uppers, finished):
        def local_core(v):
            nums = [0] * (uppers[v] + 1)
            for u in g.neighbors(v):
                nums[min(uppers[v], uppers.get(u, g.degree[u]))] += 1
                # nums[min(uppers[v], uppers[u])] += 1

            accum = 0
            for k in reversed(range(len(nums))):
                if (accum := accum + nums[k]) >= k:
                    return k

        update = False
        for v in subg.nodes:
            c_new = local_core(v)
            if c_new != uppers[v]:
                uppers[v] = c_new
                update = v not in finished

        return update

    def update_subgraph(g: nx.Graph, subg: nx.Graph, lowers, uppers):
        num_nodes = subg.number_of_nodes()
        update = False
        visited = {q}
        queue = [q]

        while queue:
            v = queue.pop()

            for u in g.neighbors(v):
                if u in visited:
                    continue

                if u not in lowers or u not in uppers:
                    lowers[u] = 0
                    uppers[u] = g.degree(u)

                if uppers[u] >= max(lowers[q], query_k):
                    visited.add(u)

                    if u not in subg.nodes:
                        update = True
                    else:
                        queue.append(u)

        # print(f"{subg.number_of_nodes()=} {len(visited)=}")

        new_subg = nx.induced_subgraph(g, visited)
        update |= len(visited) != num_nodes
        return new_subg, update

    lowers, uppers = dict(), dict()
    finished = set()

    # initialize with the 1-hop neighborhood of q
    subg = nx.induced_subgraph(g, list(g.neighbors(q)) + [q])
    lowers[q] = 0
    uppers[q] = g.degree[q]
    for u in g.neighbors(q):
        lowers[u] = 0
        uppers[u] = g.degree[u]

    # subg = g.copy()
    # lowers = {v: 0 for v in g.nodes}
    # uppers = {v: g.degree[v] for v in g.nodes}
    # frntr = set()

    all_visited = set(subg.nodes).copy()

    update = True
    while update and uppers[q] >= query_k:
        print(f"{lowers[q]=}, {uppers[q]=} {subg.number_of_nodes()=}")

        update_lower = update_lowers(subg, lowers, finished)
        update_upper = update_uppers(subg, uppers, finished)

        subg, update_subg = update_subgraph(g, subg, lowers, uppers)
        update = update_lower | update_upper | update_subg

        all_visited = all_visited.union(set(subg.nodes))

    # lowers = {v: 0 for v in g.nodes}
    # update_lowers(g, lowers)
    # uppers = {v: g.degree[v] for v in g.nodes}
    # for _ in it.count():
    #     if not update_uppers(g, uppers):
    #         break
    # for v in g.nodes:
    #     assert lowers[v] == uppers[v]

    print(f"{lowers[q]=}, {uppers[q]=}")
    print(len(all_visited), len(subg.nodes))
    return set(subg.nodes)


def main():
    g: nx.Graph = nx.read_edgelist("test/network.tsv", nodetype=int)
    with open("test/nodes.txt") as f:
        lines = [map(int, line.strip().split()) for line in f.readlines()]

    for q, k in lines:
        print(f"Getting comm for {q=} {k=}")
        cmty = get_comm(g, q, k)
        path = Path(f"test/{q}/kcore_{k}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            f.write("\n".join(map(str, cmty)))

        break


if __name__ == "__main__":
    main()
