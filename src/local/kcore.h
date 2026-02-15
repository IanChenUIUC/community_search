#ifndef KCORE_H
#define KCORE_H

#include "graph.h"
#include <algorithm>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <set>
#include <stdio.h>
#include <vector>

std::set<Vertex> get_comm(Graph *graph, Vertex q, int k);
void write_comm(char *maindir, Vertex q, Vertex k, std::set<Vertex> &comm);

#endif
