#include "graph.h"
#include "kcore.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>

// TODO: refactor to using vectors instead of sets whenever possible

Vertex update_uppers(Graph *g, Vertex cur, std::set<Vertex> &removed, std::vector<Vertex> &uppers)
{
    // the new estimate is the maximum k s.t.
    // number of neighbors(v) with core value >= k is more than k

    Vertex degree = get_degree(g, cur);
    Vertex *nbrs = neighbors(g, cur);

    // nums[k-1] = # neighbors with core (estimate) k
    std::vector<Vertex> nums(uppers[cur]);
    for (Vertex i = 0; i < degree; ++i)
    {
        if (!removed.contains(nbrs[i]))
        {
            assert(uppers[nbrs[i]] > 0);
            ++nums[std::min(uppers[cur], uppers[nbrs[i]]) - 1];
        }
    }

    Vertex k, accum = 0;
    for (k = uppers[cur]; k > 1; --k)
        if ((accum += nums[k - 1]) >= k)
            break;

    return k;
}

void update_lowers(std::map<Vertex, std::set<Vertex>> &graph, Vertex max_degree, std::set<Vertex> &candidates,
                   std::vector<Vertex> &lower)
{
    std::vector<std::set<Vertex>> queues(max_degree + 1);
    std::map<Vertex, Vertex> sub_degrees;
    for (Vertex v : candidates)
    {
        sub_degrees[v] = graph[v].size();
        queues[sub_degrees[v]].insert(v);
    }

    for (Vertex k = 1; k <= max_degree; ++k)
    {
        while (!queues[k].empty())
        {
            Vertex u = *queues[k].begin();
            queues[k].erase(u);
            lower[u] = k;

            for (Vertex v : graph[u])
            {
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

std::set<Vertex> get_comm(Graph *graph, Vertex q, Vertex k)
{
    std::vector<Vertex> lowers(graph->n_nodes), uppers(graph->n_nodes);
    std::vector<Vertex> counts(graph->n_nodes);
    std::set<Vertex> need_update;
    std::set<Vertex> candidates, removed, frontier;

    std::map<Vertex, std::set<Vertex>> sub_graph;
    Vertex max_degree = 0;

    candidates.insert(q);
    need_update.insert(q);
    uppers[q] = get_degree(graph, q);

    Vertex *q_nbrs = neighbors(graph, q);
    for (Vertex i = 0; i < get_degree(graph, q); ++i)
        frontier.insert(q_nbrs[i]);

    while (!need_update.empty())
    {
        if (uppers[q] < k)
            return {};

        // stage 0: expand the frontier
        for (Vertex cur : frontier)
            uppers[cur] = get_degree(graph, cur);

        // step 1: iterate the lower bounds
        update_lowers(sub_graph, max_degree, candidates, lowers);

        // stage 2: iterate the upper bounds
        std::set<Vertex> need_update_next;
        for (Vertex cur : need_update)
        {
            Vertex degree = get_degree(graph, cur);
            Vertex *nbrs = neighbors(graph, cur);

            Vertex c_old = uppers[cur];
            Vertex c_new = update_uppers(graph, cur, removed, uppers);
            uppers[cur] = c_new;

            counts[cur] = 0;
            for (Vertex i = 0; i < degree; ++i)
            {
                Vertex u = nbrs[i];
                if (!candidates.contains(u))
                    continue;

                if (uppers[cur] < uppers[u] && uppers[u] <= c_old)
                    --counts[u];
                if (counts[u] < uppers[u])
                    need_update_next.insert(u);
                if (uppers[cur] <= uppers[u])
                    ++counts[cur];
            }
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

                if (get_degree(graph, u) < std::min(k, lowers[q]))
                {
                    // TODO: counts and updates
                    removed.insert(u);
                    continue;
                }

                if (!candidates.contains(u) && !removed.contains(u))
                {
                    frontier_next.insert(u);
                }
                else
                {
                    sub_graph[cur].insert(u);
                    sub_graph[u].insert(cur);

                    max_degree = std::max(max_degree, sub_graph[cur].size());
                    max_degree = std::max(max_degree, sub_graph[u].size());
                }
            }
        }
        std::swap(frontier_next, frontier);

        // remove all nodes with UB[cur] < LB[q] out of candidates
        std::erase_if(candidates, [&](Vertex v) {
            if (uppers[v] >= lowers[q])
                return false;

            removed.insert(v);
            for (Vertex u : sub_graph[v])
                sub_graph[u].erase(v);
            sub_graph.erase(v);

            return true;
        });
    }

    std::set<Vertex> community;
    for (Vertex cur : candidates)
    {
        std::cout << lowers[cur] << "<=Core(" << cur << ")<=" << uppers[cur] << "\n";
        assert(lowers[cur] == uppers[cur]);
        if (lowers[cur] >= lowers[q])
            community.insert(cur);
    }

    std::cout << "Query " << q << " " << lowers[q] << "<=Core(q)<=" << uppers[q] << " Considered "
              << removed.size() + candidates.size() << " Outputted " << community.size() << "\n";

    return community;
}

void write_comm(char *maindir, Vertex q, Vertex k, std::set<Vertex> &comm)
{
    std::filesystem::path dir = std::filesystem::path(maindir) / std::to_string(q);
    std::filesystem::path path = dir / std::format("kcore_{}.txt", k);
    std::filesystem::create_directories(dir);

    std::ofstream file(path.string());
    for (Vertex v : comm)
        file << v << "\n";
}
