#ifndef GRAPH_H
#define GRAPH_H

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

typedef uint64_t Vertex;

typedef struct Graph
{
    Vertex *data;
    size_t n_nodes;
    size_t n_edges;
} Graph;

size_t get_degree(Graph *g, Vertex v);
Vertex *neighbors(Graph *g, Vertex v);

Graph *create_graph(char *adj);
void destroy_graph(Graph *g);

#endif
