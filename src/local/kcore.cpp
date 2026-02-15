#include "graph.h"
#include "kcore.h"

#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <vector>

Vertex update_uppers(Graph *g, Vertex q, std::vector<Vertex> &uppers)
{
    // the new estimate is the maximum k s.t.
    // number of neighbors(v) with core value >= k is more than k
    // fprintf(stderr, "update on q=%lu\n", q);

    Vertex degree = get_degree(g, q);
    Vertex *nbrs = neighbors(g, q);

    // nums[k-1] = # neighbors with core (estimate) k
    std::vector<Vertex> nums(degree);
    for (Vertex i = 0; i < degree; ++i)
        ++nums[std::min(uppers[q] - 1, uppers[nbrs[i]] - 1)];

    Vertex k, accum = 0;
    for (k = uppers[q]; k > 0; --k)
        if ((accum += nums[k - 1]) >= k)
            break;
    return k;
}

void update_lowers(Graph *graph, std::set<Vertex> &candidates, std::vector<Vertex> &lower)
{
    // TODO: maintain the induced degrees instead of recomputing always
    // TODO: maintain the induced graph adjacency list

    std::vector<Vertex> sub_degrees(graph->n_nodes);
    for (Vertex v : candidates)
    {
        Vertex *nbrs = neighbors(graph, v);
        Vertex degree = get_degree(graph, v);

        for (Vertex i = 0; i < degree; ++i)
            if (candidates.contains(nbrs[i]))
                sub_degrees[nbrs[i]] += 1;
    }

    Vertex max_degree = *std::max_element(sub_degrees.begin(), sub_degrees.end());
    std::vector<std::set<Vertex>> queues(max_degree + 1);
    for (Vertex v : candidates)
        queues[sub_degrees[v]].insert(v);

    // TODO: maintain the indices for each vertex in the list
    // so that we can use vectors instead of sets
    // or, use a priority queue lmao

    for (Vertex k = 1; k <= max_degree; ++k)
    {
        while (!queues[k].empty())
        {
            Vertex u = *queues[k].begin();
            queues[k].erase(u);

            lower[u] = k;

            Vertex *nbrs = neighbors(graph, u);
            Vertex degree = get_degree(graph, u);
            for (Vertex j = 0; j < degree; ++j)
            {
                Vertex v = nbrs[j];
                if (candidates.contains(v) && sub_degrees[v] > k)
                {
                    queues[sub_degrees[v]].erase(v);
                    queues[sub_degrees[v] - 1].insert(v);
                    --sub_degrees[v];
                }
            }
        }
    }
}

std::set<Vertex> get_comm(Graph *graph, Vertex q, int k)
{
    std::vector<Vertex> lowers(graph->n_nodes), uppers(graph->n_nodes);
    std::vector<Vertex> counts(graph->n_nodes);
    std::set<Vertex> need_update;
    std::set<Vertex> candidates, removed, community, frontier;

    candidates.insert(q);
    need_update.insert(q);
    uppers[q] = get_degree(graph, q);

    Vertex *q_nbrs = neighbors(graph, q);
    for (Vertex i = 0; i < get_degree(graph, q); ++i)
        frontier.insert(q_nbrs[i]);

    for (int i = 0; !need_update.empty(); ++i)
    {
        std::cout << "step i=" << i << " candidates size=" << candidates.size()
                  << " need_updates size=" << need_update.size() << "\n";

        // stage 0: expand the frontier
        for (Vertex cur : frontier)
            uppers[cur] = get_degree(graph, cur);

        // step 1: iterate the lower bounds
        update_lowers(graph, candidates, lowers);

        // stage 2: iterate the upper bounds
        std::set<Vertex> need_update_next;
        for (Vertex cur : need_update)
        {
            Vertex degree = get_degree(graph, cur);
            Vertex *nbrs = neighbors(graph, cur);

            Vertex c_old = uppers[cur];
            Vertex c_new = update_uppers(graph, cur, uppers);
            uppers[cur] = c_new;

            for (Vertex i = 0; i < degree; ++i)
            {
                if (!candidates.contains(nbrs[i]))
                    continue;

                if (uppers[cur] < uppers[nbrs[i]] && uppers[nbrs[i]] <= c_old)
                    --counts[nbrs[i]];
                if (counts[nbrs[i]] < uppers[nbrs[i]])
                    need_update_next.insert(nbrs[i]);
            }

            counts[cur] = 0;
            for (Vertex i = 0; i < degree; ++i)
                if (candidates.contains(nbrs[i]) && uppers[cur] <= uppers[nbrs[i]])
                    ++counts[cur];
        }
        std::swap(need_update_next, need_update);

        // step 3: figure out the updates for next iteration
        std::set<Vertex> frontier_next;
        for (Vertex cur : frontier)
        {
            candidates.insert(cur);
            need_update.insert(cur);

            Vertex degree = get_degree(graph, cur);
            Vertex *nbrs = neighbors(graph, cur);
            for (Vertex i = 0; i < degree; ++i)
            {
                Vertex u = nbrs[i];
                if (!candidates.contains(u) && !removed.contains(u) && get_degree(graph, u) >= lowers[q])
                    frontier_next.insert(u);
            }
        }
        std::swap(frontier_next, frontier);

        // remove all nodes with UB[cur] < LB[q] out of candidates
        for (Vertex cur : candidates)
            if (uppers[cur] < lowers[q])
                removed.insert(cur);
        std::erase_if(candidates, [&](Vertex cur) { return uppers[cur] < lowers[q]; });
    }
    return candidates;
}

void write_comm(char *maindir, Vertex q, Vertex k, std::set<Vertex> &comm)
{
    std::filesystem::path dir = std::filesystem::path(maindir) / std::to_string(q);
    std::filesystem::path path = dir / std::format("kcore_{}.txt", k);
    if (!std::filesystem::create_directories(dir))
    {
        fprintf(stderr, "Could not create directory\n");
        exit(1);
    }

    std::ofstream file(path.string());
    for (Vertex v : comm)
        file << v << "\n";
}
