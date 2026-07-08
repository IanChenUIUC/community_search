#ifndef KCORE_H
#define KCORE_H

#include "graph.h"
#include <set>

std::set<Vertex> get_comm(Graph *graph, Vertex q, Vertex k);
void write_comm(char *maindir, Vertex q, Vertex k, std::set<Vertex> &comm);

#endif
