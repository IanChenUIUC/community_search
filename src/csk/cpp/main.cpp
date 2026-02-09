#include "graph_IO.h"
#include <iostream>
#include <string>

using namespace std;

void indexinputSearch(int argc, char *argv[], bool nodelist)
{
    char *maindir = argv[2];
    char *graphdata = argv[3];
    char *indexfile = argv[4];
    char *infile = argv[5];

    Readin_Graph(graphdata);

    coreness = new int[n];
    memset(coreness, 0, n * sizeof(int));

    FILE *findex = fopen(indexfile, "r");
    if (findex == NULL)
    {
        cout << "Failed to find index file" << endl;
        return;
    }

    int vi, cor;
    while (fscanf(findex, "%d %d", &vi, &cor) != EOF)
        coreness[vi] = cor;

    FILE *fin = fopen(infile, "r");
    if (fin == NULL)
    {
        cout << "Failed to find node" << endl;
        return;
    }

    int x, y;
    while (fscanf(fin, "%d %d", &x, &y) != EOF)
    {
        cout << "community search: id= " << x << ", k=" << y << endl;
        int v = x;
        int k = y;

        kcoreCommunitySearch(v, k);

        if (nodelist)
            PrintkcoreCommunity_Nodes(maindir, v, k);
        else
            PrintkcoreCommunity(maindir, v, k);

        comm.clear();
        kcore_visited_nodes.clear();
    }
}

void buildcoreIndex(int argc, char *argv[])
{
    cout << "Building index.." << endl;
    char *maindir = argv[2];
    char *graphdata = argv[3];

    string genfile = std::string(maindir) + "kcore_index.txt";
    ofstream gout(genfile, ios::out | ios::trunc | ios::binary);

    Readin_Graph(graphdata);
    core_decompostion();

    for (int i = 0; i < n; ++i)
        gout << i << " " << coreness[i] << endl;
    gout.close();
}

void printUsage(char *name)
{
    fprintf(stderr, "Usage: %s [COMMAND] [ARGS]...\n\n", name);

    fprintf(stderr, "%s -i maindir edgelist\n", name);
    fprintf(stderr, "Build k-core Index: \n\t"
                    "@edgelist is two column (tab separated) unweighted graph. "
                    "The node ids are from 0 to N-1 (memory will be allocated for up to vertex_id_max nodes). \n\t"
                    "@maindir is the directory where the output @maindir/kcore_index.txt will be created. "
                    "maindir should end with a '/'. \n\t"
                    "kcore_index is a two column file (space separated) with node and its core number\n\n");

    fprintf(stderr, "%s -s maindir edgelist indexfile nodes --nodelist\n", name);
    fprintf(stderr, "Search k-core Communities: \n\t"
                    "@edgelist [see comments above] \n\t"
                    "@maindir [see comments above] \n\t"
                    "@indexfile is the output of the -i command above. \n\t"
                    "@nodes is a two column file of node and minimum core number.\n\t"
                    "--nodelist is an option that outputs the nodes (intead of edges) of the community.\n\n");
}

int main(int argc, char *argv[])
{
    if (argc == 4 && strcmp(argv[1], "-i") == 0)
        buildcoreIndex(argc, argv);
    else if (argc == 6 && strcmp(argv[1], "-s") == 0)
        indexinputSearch(argc, argv, false);
    else if (argc == 7 && strcmp(argv[1], "-s") == 0 && strcmp(argv[6], "--nodelist") == 0)
        indexinputSearch(argc, argv, true);
    else
        printUsage(argv[0]);
}
