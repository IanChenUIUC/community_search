#include "graph.h"
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

size_t get_degree(Graph *g, Vertex v)
{
    size_t start, end;

    start = g->data[v];
    end = (v == g->n_nodes - 1) ? g->n_nodes + g->n_edges : g->data[v + 1];
    return end - start;
}

Vertex *neighbors(Graph *g, Vertex v)
{
    return g->data + g->data[v];
}

Graph *create_graph(char *adj)
{
    int filefd;
    off_t size;
    Vertex *data = NULL;
    Graph *graph = NULL;

    if ((filefd = open(adj, O_RDONLY)) == -1)
        goto cleanup1;
    if ((size = lseek(filefd, 0, SEEK_END)) == -1)
        goto cleanup2;
    if ((data = (Vertex *)mmap(NULL, size, PROT_READ, MAP_PRIVATE, filefd, 0)) == MAP_FAILED)
        goto cleanup2;
    if ((graph = (Graph *)calloc(1, sizeof(*graph))) == NULL)
        goto cleanup2;

    graph->data = data;
    graph->n_nodes = graph->data[0];
    graph->n_edges = (size / sizeof(Vertex)) - graph->n_nodes;

cleanup2:
    close(filefd);
cleanup1:
    return graph;
}

void destroy_graph(Graph *g)
{
    munmap(g->data, sizeof(Vertex) * (g->n_edges + g->n_nodes));
    free(g);
}
