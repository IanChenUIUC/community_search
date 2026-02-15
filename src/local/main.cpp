#include "graph.h"
#include "kcore.h"

#include <stdio.h>

#define diep(m)                                                                                                        \
    perror(m);                                                                                                         \
    return;

void print_usage(char *name)
{
    fprintf(stderr, "Usage: %s [COMMAND] [ARGS]...\n\n", name);

    fprintf(stderr, "%s -s maindir graph nodes\n", name);
    fprintf(stderr, "Search k-core Communities: \n\t"
                    "graph the graph format should be a binary from the converter \n\t"
                    "@maindir this is the directory to write the output @maindir/{node}/kcore_k{k}.txt \n\t"
                    "@nodes is a two column file of node and minimum core number.\n\n");
}

void local_kcore_search(char *maindir, char *graphfile, char *nodesfile)
{
    Graph *graph = NULL;
    FILE *nodes = NULL;
    Vertex q;
    int k;

    if ((nodes = fopen(nodesfile, "r")) == NULL)
    {
        fprintf(stderr, "nodes=%p\n", nodes);
        diep("could not find nodes");
    }
    if ((graph = create_graph(graphfile)) == NULL)
    {
        fprintf(stderr, "graph=%p\n", graph);
        diep("could not create graph");
    }

    while (fscanf(nodes, "%lu %d", &q, &k) != EOF)
    {
        fprintf(stdout, "Running k-core on query=%lu min_k=%d\n", q, k);

        auto comm = get_comm(graph, q, k);
        write_comm(maindir, q, k, comm);
    }

    destroy_graph(graph);
}

int main(int argc, char **argv)
{
    if (argc == 5 && strcmp(argv[1], "-s") == 0)
        local_kcore_search(argv[2], argv[3], argv[4]);
    else
        print_usage(argv[0]);
}
