#include "graph.h"
#include <fstream>
#include <unordered_set>
typedef struct endpoint
{
    int u;
    int eid;

} endpoint;

vector<endpoint> *g;
set<int> compset;
int n, m, dmax, maxsup, maxtruss;
int Tsize, K;

vector<int> *bin_sort;
vector<vector<int>> comm;
map<int, int> communitymapping;
int communityID;

int *nodeecc;

unordered_set<int> sparseEdges;

int test_num = 0;
// flag[x]=true, the score of v has been computed.
bool *flag;
// conp[x] as the score of x
int *conp;
int *mv;
vector<endpoint> *inde;

int *coreness;
vector<int> qnodes;
int *nodetruss;

unordered_set<int> kcore_visited_nodes;

// the sum of Top-k values
int total_comp = 0;

set<int> removedEdges;

typedef struct Edgetype
{
    int x, y, truss;
} Edgetype;

Edgetype *eg;
int *sup, *truss;

map<int, vector<endpoint>> *t;

map<pair<int, int>, int> eset;
vector<int> *L;

string graphname;

bool *del;

typedef struct Score
{
    int id;
    int score;
} Score;

bool operator<(const Score &t1, const Score &t2)
{
    return t1.score > t2.score;
}

typedef struct Setree
{
    int father, rank, count;
    int cost;
} Setree;

typedef struct SuperNode
{
    int root_id;
    vector<int> internal_nodes;
    int truss;
} SuperNode;

typedef struct AdvancedInde
{
    // vector<Edgetype> superedges;
    map<int, SuperNode> supernodes;
    map<int, vector<int>> nodebin;
    map<int, vector<Edgetype>> edgebin;
    int knmax;
    int kmmax;
    // vector<int> internal_nodes;
} AdvancedInde;

AdvancedInde *adi;

typedef struct Index
{
    double upper_bound;
    map<int, Setree> t;

    int change;

    void init(int id)
    {
        t[0].father = 0;
        t[0].rank = -1;
        t[0].count = 0;
        t[0].cost = 0;

        for (int i = 0; i < g[id].size(); i++)
        {
            int x = g[id][i].u;
            t[x].father = x;
            t[x].rank = 0;
            t[x].count = 1;
            t[x].cost = min(g[x].size(), g[id].size());
        }
    }

    void initSubgraph(int nid)
    {
        t[0].father = 0;
        t[0].rank = -1;
        t[0].count = 0;
        t[0].cost = 0;
        for (int i = 1; i < nid; ++i)
        {
            t[i].father = i;
            t[i].rank = 0;
            t[i].count = 1;
            t[i].cost = 0;
        }
    }

    int Find_Father(int x)
    {
        if (x != t[x].father)
        {
            t[x].father = Find_Father(t[x].father);
            return t[x].father;
        }
        return t[x].father;
    }
    void Union(int x, int y)
    {
        int fx, fy;
        fx = Find_Father(x);
        fy = Find_Father(y);

        if (fx == fy)
            return;
        if (t[fx].rank > t[fy].rank)
        {
            t[fy].father = fx;
            t[fx].count += t[fy].count;
            t[fx].cost += t[fy].cost;
        }
        else
        {
            t[fx].father = fy;
            t[fy].count += t[fx].count;
            t[fy].cost += t[fx].cost;

            if (t[fx].rank == t[fy].rank)
            {
                t[fy].rank++;
            }
        }
        change = 1;
    }
    void Single_Father(int x)
    {
        t[x].father = 0;
        t[0].count++;
        change = 1;
    }
} Index;

// Union_Find_Isolate data structure(AB)
Index *Set;

priority_queue<Score, vector<Score>> result;

void truss_decompostion()
{
    // ofstream sout(outfile,ios::out|ios::app|ios::binary);
    clock_t start_t, end_t;
    maxsup = 0;
    maxtruss = 0;
    L = new vector<int>[dmax + 1];
    inde = new vector<endpoint>[n];
    nodetruss = new int[n];

    int totalSup = 0;

    start_t = clock();

    for (int i = 0; i < n; i++)
    {
        int x = i;
        for (int j = 0; j < g[i].size(); j++)
        {
            int y = g[i][j].u;
            if (x < y)
            {
                int eid = g[i][j].eid;
                sup[eid] = 0;
                int u, v;
                u = y;
                v = x;
                if (g[x].size() < g[y].size())
                {
                    u = x;
                    v = y;
                }
                for (int k = 0; k < g[u].size(); k++)
                {
                    int w = g[u][k].u;
                    if (eset.find(pair<int, int>(v, w)) != eset.end())
                    {
                        sup[eid]++;
                        totalSup++;
                    }
                }
                // printf("%d %d id=%d, sup=%d\n", x, y, eid, sup[eid]);
                if ((sup[eid]) > maxsup)
                {
                    maxsup = sup[eid];
                }
                L[sup[eid]].push_back(eid);
            }
        }
    }
    end_t = clock();

    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // printf("Triangle Count: =%.3lf\n", total_t);

    start_t = clock();
    del = new bool[m];
    for (int i = 0; i < m; i++)
    {
        del[i] = false;
    }
    //	if(maxsup>Tsize-2){
    //		maxsup=Tsize-2;
    //	}
    unordered_set<int> qn;
    for (int k = 0; k <= maxsup; k++)
    {
        for (int i = 0; i < L[k].size(); i++)
        {
            int eid = L[k][i];
            if (del[eid] == true)
                continue;
            if ((k + 2) > maxtruss)
                maxtruss = k + 2;
            int x = eg[eid].x;
            int y = eg[eid].y;
            truss[eid] = k + 2;

            nodetruss[x] = k + 2;
            nodetruss[y] = k + 2;
            //			if (k+2>=10)
            //            {
            //                if(qnodes.size()<=400)
            //                {
            //                    if(qn.find(x)==qn.end())
            //                    {
            //                        qn.insert(x);
            //                        qnodes.push_back(x);
            //                    }
            //                    if(qn.find(y)==qn.end())
            //                    {
            //                        qn.insert(y);
            //                        qnodes.push_back(y);
            //                    }
            //                }
            //
            //            }

            del[eid] = true;
            sup[eid]--;
            int u, v;
            endpoint a, b;

            /*if (x<y)
            {
              a.u = y;
              a.eid = truss[eid];
              inde[x].push_back(a);
            }
            else
            {
              b.u = x;
              b.eid = truss[eid];
              inde[y].push_back(b);
            }*/
            a.u = y;
            a.eid = truss[eid];
            inde[x].push_back(a);
            b.u = x;
            b.eid = truss[eid];
            inde[y].push_back(b);

            u = y;
            v = x;
            if (g[x].size() < g[y].size())
            {
                u = x;
                v = y;
            }
            for (int j = 0; j < g[u].size(); j++)
            {
                int w = g[u][j].u;
                int temp = g[u][j].eid;
                if (del[temp] == true)
                    continue;

                if (eset.find(pair<int, int>(v, w)) != eset.end())
                {
                    temp = eset[pair<int, int>(v, w)];
                    if (del[temp] == true)
                        continue;

                    if (sup[temp] > k)
                    {
                        sup[temp]--;
                        L[sup[temp]].push_back(temp);
                    }
                    temp = g[u][j].eid;
                    if (sup[temp] > k)
                    {
                        sup[temp]--;
                        L[sup[temp]].push_back(temp);
                    }
                }
            }
            // printf("%d %d id=%d, truss=%d\n", x, y, eid, truss[eid]);
        }
    }
    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // sout<<"running time: "<<total_t<<"s"<<endl;
    //	printf("Truss Decomposition: =%.3lf\n", total_t);
    printf("dmax=%d, maxtruss=%d \n", dmax, maxtruss);
    // cout<<"triangle number: "<<totalSup/3<<", total support: "<<totalSup<<endl;
    delete[] L;
    delete[] sup;
}

void truss_decompostion_K(int threshold_K)
{
    // ofstream sout(outfile,ios::out|ios::app|ios::binary);
    clock_t start_t, end_t;
    maxsup = 0;
    maxtruss = 0;
    L = new vector<int>[dmax + 1];
    // inde = new vector <endpoint> [n+1];
    nodetruss = new int[n];

    int totalSup = 0;

    start_t = clock();

    for (int i = 0; i < n; i++)
    {
        int x = i;
        for (int j = 0; j < g[i].size(); j++)
        {
            int y = g[i][j].u;
            if (x < y)
            {
                int eid = g[i][j].eid;
                sup[eid] = 0;
                int u, v;
                u = y;
                v = x;
                if (g[x].size() < g[y].size())
                {
                    u = x;
                    v = y;
                }
                for (int k = 0; k < g[u].size(); k++)
                {
                    int w = g[u][k].u;
                    if (eset.find(pair<int, int>(v, w)) != eset.end())
                    {
                        sup[eid]++;
                        totalSup++;
                    }
                }
                // printf("%d %d id=%d, sup=%d\n", x, y, eid, sup[eid]);
                if ((sup[eid]) > maxsup)
                {
                    maxsup = sup[eid];
                }
                L[sup[eid]].push_back(eid);
            }
        }
    }
    end_t = clock();

    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // printf("Triangle Count: =%.3lf\n", total_t);

    start_t = clock();
    del = new bool[m];
    for (int i = 0; i < m; i++)
    {
        del[i] = false;
    }
    //	if(maxsup>Tsize-2){
    //		maxsup=Tsize-2;
    //	}
    unordered_set<int> qn;
    for (int k = 0; k <= maxsup; k++)
    {
        if (k + 2 > threshold_K)
        {
            break;
        }
        for (int i = 0; i < L[k].size(); i++)
        {
            int eid = L[k][i];
            if (del[eid] == true)
                continue;
            if ((k + 2) > maxtruss)
                maxtruss = k + 2;
            int x = eg[eid].x;
            int y = eg[eid].y;
            truss[eid] = k + 2;

            nodetruss[x] = k + 2;
            nodetruss[y] = k + 2;
            //			if (k+2>=10)
            //            {
            //                if(qnodes.size()<=400)
            //                {
            //                    if(qn.find(x)==qn.end())
            //                    {
            //                        qn.insert(x);
            //                        qnodes.push_back(x);
            //                    }
            //                    if(qn.find(y)==qn.end())
            //                    {
            //                        qn.insert(y);
            //                        qnodes.push_back(y);
            //                    }
            //                }
            //
            //            }

            del[eid] = true;
            sup[eid]--;
            int u, v;
            endpoint a, b;

            /*if (x<y)
            {
              a.u = y;
              a.eid = truss[eid];
              inde[x].push_back(a);
            }
            else
            {
              b.u = x;
              b.eid = truss[eid];
              inde[y].push_back(b);
            }*/
            //			a.u = y;
            //			a.eid = truss[eid];
            //			inde[x].push_back(a);
            //			b.u = x;
            //			b.eid = truss[eid];
            //			inde[y].push_back(b);

            u = y;
            v = x;
            if (g[x].size() < g[y].size())
            {
                u = x;
                v = y;
            }
            for (int j = 0; j < g[u].size(); j++)
            {
                int w = g[u][j].u;
                int temp = g[u][j].eid;
                if (del[temp] == true)
                    continue;

                if (eset.find(pair<int, int>(v, w)) != eset.end())
                {
                    temp = eset[pair<int, int>(v, w)];
                    if (del[temp] == true)
                        continue;

                    if (sup[temp] > k)
                    {
                        sup[temp]--;
                        L[sup[temp]].push_back(temp);
                    }
                    temp = g[u][j].eid;
                    if (sup[temp] > k)
                    {
                        sup[temp]--;
                        L[sup[temp]].push_back(temp);
                    }
                }
            }
            // printf("%d %d id=%d, truss=%d\n", x, y, eid, truss[eid]);
        }
    }
    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // sout<<"running time: "<<total_t<<"s"<<endl;
    //	printf("Truss Decomposition: =%.3lf\n", total_t);
    // printf("dmax=%d, maxtruss=%d \n", dmax, maxtruss);
    // cout<<"triangle number: "<<totalSup/3<<", total support: "<<totalSup<<endl;
    delete[] L;
    //	delete [] sup;
}

void core_decompostion()
{
    // cout<<g[vertex].size()<<endl;

    clock_t start_t, end_t;
    maxsup = 0;
    maxtruss = 0;
    L = new vector<int>[n];

    sup = new int[m];
    truss = new int[m];

    int *degg = new int[n];
    int maxdeg = 0;
    int maxcore = 0;

    //    for (int i1 = 0; i1 < edgeid; ++i1) {
    //        sup[i1]=0;
    //    }

    removedEdges.clear();

    start_t = clock();

    for (int i = 0; i < n; i++)
    {
        int x = i;
        // cout<<"x:"<<x<<"\n";
        degg[x] = g[x].size();
        // cout<<"degree_x:"<<degg[x]<<"\n";
        if (maxdeg < degg[x])
        {
            maxdeg = degg[x];
        }
        L[degg[x]].push_back(x);
    }

    //    for (int i=0; i<g[vertex].size(); i++)
    //    {
    //        int x = g[vertex][i].u;
    //        for (int j=0; j<g[x].size(); j++)
    //        {
    //            int y = g[x][j].u;
    //            if (eset.find(pair<int, int>(vertex, y)) == eset.end())
    //            {
    //                continue;
    //                //cout<<u<<"  "<<v<<"  "<<w<<" "<<sup[eid]<<endl;
    //            }
    //            if (x <y )
    //            {
    //                int eid = g[x][j].eid;
    //                sup[eid] = 0;
    //                int u, v;
    //                u = y;
    //                v = x;
    //                if (g[x].size() <g[y].size())
    //                {
    //                    u  = x;
    //                    v = y;
    //                }
    //                for (int k=0; k<g[u].size(); k++)
    //                {
    //                    int w = g[u][k].u;
    //                    if (eset.find(pair<int, int>(v, w)) != eset.end())
    //                    {
    //
    //                        if (eset.find(pair<int, int>(vertex, u)) != eset.end() && eset.find(pair<int, int>(vertex,
    //                        v)) != eset.end() && eset.find(pair<int, int>(vertex, w)) != eset.end() )
    //                        {
    //                            sup[eid]++;
    //                            //cout<<u<<"  "<<v<<"  "<<w<<" "<<sup[eid]<<endl;
    //                        }
    //                        //cout<<eid<<endl;
    //                    }
    //                }
    //                //printf("%d %d id=%d, sup=%d\n", x, y, eid, sup[eid]);
    //                if ((sup[eid]) > maxsup)
    //                {
    //                    maxsup = sup[eid];
    //                }
    //                L[sup[eid]].push_back(eid);
    //            }
    //        }
    //    }
    end_t = clock();

    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // printf("Triangle Count: =%.3lf\n", total_t);

    //    for (int i=0;i<m/2;i++)
    //    {
    //        cout<<sup[i]<<endl;
    //    }

    start_t = clock();
    del = new bool[n];
    coreness = new int[n];
    for (int i = 0; i < n; i++)
    {
        del[i] = false;
    }
    int kmax = 0;
    //	vector<Edgetype> *gx=new vector<Edgetype> [maxsup+3];
    int md = maxdeg;
    //	if(maxdeg>Tsize){
    //		maxdeg=Tsize;
    //	}
    for (int k = 0; k <= maxdeg; k++)
    {
        for (int i = 0; i < L[k].size(); i++)
        {
            // cout<<"k, L[k] size:"<<k<<","<<L[k].size()<<"\n";
            int x = L[k][i];
            // cout<<"x:"<<x<<"\n";
            if (del[x] == true)
            {
                // cout<<"Deleted:"<<x<<"\n";
                continue;
            }
            coreness[x] = k;
            // cout<<"coreness:"<<coreness[x]<<"\n";
            if (maxcore < coreness[x])
            {
                maxcore = coreness[x];
                // cout<<"maxcore:"<<maxcore<<"\n";
            }
            del[x] = true;
            for (int j = 0; j < g[x].size(); ++j)
            {
                int y = g[x][j].u;
                // cout<<"y:"<<y<<"\n";

                if (del[y] == true)
                {
                    // cout<<"Deleted:"<<y<<"\n";
                    continue;
                }

                int realx = x;
                int realy = y;

                int delEid = eset[pair<int, int>(realx, realy)];

                if (coreness[x] < Tsize)
                {
                    // cout<<"Inside this";
                    removedEdges.insert(delEid);
                }

                int temp = y;

                if (degg[temp] > k)
                {
                    degg[temp]--;
                    // cout<<"Reduced:"<<temp<<","<<degg[temp]<<"\n";
                    L[degg[temp]].push_back(temp);
                }
            }
            // printf("%d %d id=%d, truss=%d\n", x, y, eid, truss[eid]);
        }
    }

    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // printf("Truss Decomposition: =%.3lf\n", total_t);
    // cout<<"id="<<vertex<<", ";
    printf("maxdegree=%d, maxcore=%d \n", md, maxcore);
    delete[] L;
    delete[] inde;
    delete[] del;
    delete[] sup;
    delete[] truss;
    delete[] degg;
}

void core_decompostion_K(int treshold_K)
{
    // cout<<g[vertex].size()<<endl;

    clock_t start_t, end_t;
    maxsup = 0;
    maxtruss = 0;
    L = new vector<int>[n];

    //	sup=new int[m];
    //	truss=new int[m];

    int *degg = new int[n];
    int maxdeg = 0;
    int maxcore = 0;

    //    for (int i1 = 0; i1 < edgeid; ++i1) {
    //        sup[i1]=0;
    //    }

    removedEdges.clear();

    start_t = clock();

    for (int i = 0; i < n; i++)
    {
        int x = i;
        degg[x] = g[x].size();
        if (maxdeg < degg[x])
        {
            maxdeg = degg[x];
        }
        L[degg[x]].push_back(x);
    }

    //    for (int i=0; i<g[vertex].size(); i++)
    //    {
    //        int x = g[vertex][i].u;
    //        for (int j=0; j<g[x].size(); j++)
    //        {
    //            int y = g[x][j].u;
    //            if (eset.find(pair<int, int>(vertex, y)) == eset.end())
    //            {
    //                continue;
    //                //cout<<u<<"  "<<v<<"  "<<w<<" "<<sup[eid]<<endl;
    //            }
    //            if (x <y )
    //            {
    //                int eid = g[x][j].eid;
    //                sup[eid] = 0;
    //                int u, v;
    //                u = y;
    //                v = x;
    //                if (g[x].size() <g[y].size())
    //                {
    //                    u  = x;
    //                    v = y;
    //                }
    //                for (int k=0; k<g[u].size(); k++)
    //                {
    //                    int w = g[u][k].u;
    //                    if (eset.find(pair<int, int>(v, w)) != eset.end())
    //                    {
    //
    //                        if (eset.find(pair<int, int>(vertex, u)) != eset.end() && eset.find(pair<int, int>(vertex,
    //                        v)) != eset.end() && eset.find(pair<int, int>(vertex, w)) != eset.end() )
    //                        {
    //                            sup[eid]++;
    //                            //cout<<u<<"  "<<v<<"  "<<w<<" "<<sup[eid]<<endl;
    //                        }
    //                        //cout<<eid<<endl;
    //                    }
    //                }
    //                //printf("%d %d id=%d, sup=%d\n", x, y, eid, sup[eid]);
    //                if ((sup[eid]) > maxsup)
    //                {
    //                    maxsup = sup[eid];
    //                }
    //                L[sup[eid]].push_back(eid);
    //            }
    //        }
    //    }
    end_t = clock();

    double total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // printf("Triangle Count: =%.3lf\n", total_t);

    //    for (int i=0;i<m/2;i++)
    //    {
    //        cout<<sup[i]<<endl;
    //    }

    start_t = clock();
    del = new bool[n];
    coreness = new int[n];
    memset(coreness, treshold_K, (n) * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        del[i] = false;
    }
    int kmax = 0;
    //	vector<Edgetype> *gx=new vector<Edgetype> [maxsup+3];
    int md = maxdeg;
    //	if(maxdeg>Tsize){
    //		maxdeg=Tsize;
    //	}
    for (int k = 0; k <= maxdeg; k++)
    {
        if (k > treshold_K)
        {
            break;
        }
        for (int i = 0; i < L[k].size(); i++)
        {
            int x = L[k][i];
            if (del[x] == true)
            {
                continue;
            }
            coreness[x] = k;
            if (maxcore < coreness[x])
            {
                maxcore = coreness[x];
            }
            del[x] = true;
            for (int j = 0; j < g[x].size(); ++j)
            {
                int y = g[x][j].u;

                if (del[y] == true)
                {
                    continue;
                }

                int realx = x;
                int realy = y;

                int delEid = eset[pair<int, int>(realx, realy)];

                if (coreness[x] < Tsize)
                {
                    removedEdges.insert(delEid);
                }

                int temp = y;

                if (degg[temp] > k)
                {
                    degg[temp]--;
                    L[degg[temp]].push_back(temp);
                }
            }
            // printf("%d %d id=%d, truss=%d\n", x, y, eid, truss[eid]);
        }
    }

    end_t = clock();
    total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    // printf("Truss Decomposition: =%.3lf\n", total_t);
    // cout<<"id="<<vertex<<", ";
    //	printf("maxdegree=%d, maxcore=%d \n", md, maxcore);
    delete[] L;
    //	delete [] inde;
    delete[] del;
    delete[] degg;
}

void ktrussCommunitySearch(int v, int k)
{
    unordered_set<int> visited_edges;
    int l = 0;
    for (int i = 0; i < g[v].size(); i++)
    {
        int u = g[v][i].u;
        int neid = g[v][i].eid;
        if (truss[neid] >= k && visited_edges.find(neid) == visited_edges.end())
        {
            l++;
            vector<int> cl;
            queue<int> Q;
            Q.push(neid);
            visited_edges.insert(neid);
            while (!Q.empty())
            {
                int cureid = Q.front();
                Q.pop();
                cl.push_back(cureid);

                int p = eg[cureid].x;
                int q = eg[cureid].y;
                int x = p;
                int y = q;
                if (g[p].size() > g[q].size())
                {
                    x = q;
                    y = p;
                }
                for (int j = 0; j < g[x].size(); j++)
                {
                    int z = g[x][j].u;
                    if (eset.find(pair<int, int>(y, z)) != eset.end())
                    {
                        int xzeid = eset[pair<int, int>(x, z)];
                        int yzeid = eset[pair<int, int>(y, z)];
                        if (truss[xzeid] >= k && truss[yzeid] >= k)
                        {
                            if (visited_edges.find(xzeid) == visited_edges.end())
                            {
                                visited_edges.insert(xzeid);
                                Q.push(xzeid);
                            }

                            if (visited_edges.find(yzeid) == visited_edges.end())
                            {
                                Q.push(yzeid);
                                visited_edges.insert(yzeid);
                            }
                        }
                    }
                }
            }

            comm.push_back(cl);
        }
    }
}

void kcoreCommunitySearch(int v, int k)
{
    //	if (communitymapping.find(v)!=communitymapping.end())
    //	{
    //		return;
    //	}
    //	communityID++;
    unordered_set<int> visited_edges;

    queue<int> Q;
    // Q.push(v);
    // visited_nodes.insert(v);
    //	communitymapping[v]=communityID;

    // VKP,28th Aug: Query node should have a coreness >=k
    if (coreness[v] >= k)
    {
        Q.push(v);
        kcore_visited_nodes.insert(v);
    }
    else
    {
        printf("Coreness of %d is %d less than k=%d\n", v, coreness[v], k);
    }
    vector<int> cl;

    while (!Q.empty())
    {
        int x = Q.front();
        Q.pop();

        for (int i = 0; i < g[x].size(); ++i)
        {
            int y = g[x][i].u;
            int cureid = g[x][i].eid;
            if (coreness[y] >= k)
            {
                if (visited_edges.find(cureid) == visited_edges.end())
                {
                    cl.push_back(cureid);
                    visited_edges.insert(cureid);
                }
                if (kcore_visited_nodes.find(y) == kcore_visited_nodes.end())
                {
                    kcore_visited_nodes.insert(y);
                    Q.push(y);

                    // communitymapping[y]=communityID;
                }
            }
        }
    }

    comm.push_back(cl);
    cout << "Added cl to comm:" << v << endl;
}

void maxKtrussCommunitySearch(int v, int k)
{
    if (communitymapping.find(v) != communitymapping.end())
    {
        return;
    }
    communityID++;
    unordered_set<int> visited_nodes;
    unordered_set<int> visited_edges;

    queue<int> Q;
    Q.push(v);
    visited_nodes.insert(v);
    communitymapping[v] = communityID;
    vector<int> cl;

    while (!Q.empty())
    {
        int x = Q.front();
        Q.pop();

        for (int i = 0; i < g[x].size(); ++i)
        {
            int y = g[x][i].u;
            int cureid = g[x][i].eid;
            if (nodetruss[y] >= 6)
            {
                if (visited_edges.find(cureid) == visited_edges.end())
                {
                    cl.push_back(cureid);
                    visited_edges.insert(cureid);
                }
                if (visited_nodes.find(y) == visited_nodes.end())
                {
                    visited_nodes.insert(y);
                    Q.push(y);
                    communitymapping[y] = communityID;
                }
            }
        }
    }

    comm.push_back(cl);
}
