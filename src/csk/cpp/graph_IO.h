#include "truss.h"
#include <sys/stat.h>
// Read Input Graph Dataset  and store in g.
void Readin_Graph(char *str)
{
    int linecount = 0;
    FILE *fin = fopen(str, "r");
    if (fin == NULL)
    {
        cout << "could not open file" << endl;
        exit(1);
    }

    int x, y, z;

    n = m = 0;
    dmax = 0;
    while (fscanf(fin, "%d\t%d", &x, &y) != EOF)
    {
        n = max(n, x);
        n = max(n, y);
    }
    n++;
    fclose(fin);
    g = new vector<endpoint>[n + 1];

    // Vertex id starting from 1.
    fin = fopen(str, "r");
    endpoint a, b;
    Edgetype e;
    int eid = 0;

    while (fscanf(fin, "%d\t%d", &x, &y) != EOF)
    {
        linecount++;
        if (linecount % 10000000 == 0)
        {
            cout << "processed edge no: " << linecount << endl;
        }
        if (x == y)
        {
            continue;
        }
        if (y < x)
            swap(x, y);
        if (x < y)
        {
            if (eset.find(pair<int, int>(x, y)) == eset.end())
            {

                eset[pair<int, int>(x, y)] = eid;
                eset[pair<int, int>(y, x)] = eid;
                a.u = y;
                a.eid = eid;
                g[x].push_back(a);
                b.u = x;
                b.eid = eid;
                g[y].push_back(b);

                eid++;
                m += 2;
            }
        }
        n = max(y, n);
    }

    eg = new Edgetype[eid];
    sup = new int[eid];
    truss = new int[eid];

    for (int i = 0; i < n; i++)
    {
        int x = i;
        for (int j = 0; j < g[i].size(); j++)
        {
            int y = g[i][j].u;
            int z = g[i][j].eid;
            if (x < y)
            {
                e.x = x;
                e.y = y;
                eg[z] = e;
                // printf("%d %d id=%d\n", x, y, z);
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        dmax = max(dmax, (int)g[i].size());
    }
    //	printf("n = %d, m = %d, dmax = %d, eid=%d\n", n, m, dmax, eid);
    fclose(fin);
}

// Print the sum of Top-k values, Running Time(s), the number of vertices that have computed score.
void Print_Ans(char *method_name, double timeuse)
{
    Score temp;
    int sum = 0;
    set<int> trresult;
    map<int, vector<int>> origin;
    map<int, vector<int>> real;
    set<int> real_result;

    int x;
    //	FILE *fin = fopen("Livejournal/gowalla_propagate_4_100_atn_core.txt", "r");
    //	while(fscanf(fin, "%d", &x)!=EOF)
    //	{
    //		real_result.insert(x);
    //	}

    // ofstream sout("fbanon/fbanon_4_100_truss.txt",ios::out|ios::trunc|ios::binary);
    while (!result.empty())
    {

        temp = result.top();
        result.pop();
        sum++;
        sum = 0;
        total_comp += (int)temp.score;
        //		for (int i=0; i<g[temp.id].size(); i++)
        //		{
        //			sum += g[g[temp.id][i].u].size();
        //		}
        // sout<<temp.id<<endl;
        //		trresult.insert(temp.id);
        printf("id=%d deg=%d conp=%d\n", temp.id, 0, temp.score);

        //		origin[temp.score].push_back(temp.id);
        //
        //		if (real_result.find(temp.id)!=real_result.end())
        //		{
        //			real[temp.score].push_back(temp.id);
        //		}
    }

    //	map<int,vector<int>>::iterator it;
    //	for (it=origin.begin();it!=origin.end();it++)
    //	{
    //		int score=it->first;
    //		int number=it->second.size();
    //		int real_num=real[score].size();
    //
    //		cout<<score<<","<<number<<","<<real_num<<endl;
    //	}
    //	sout.close();
    // cout<<"total element in Queue: "<<sum<<endl;

    //	vector<int> *degsort=new vector<int> [n+1];
    //	ofstream sdout("astroPh_deg_100.txt",ios::out|ios::trunc|ios::binary);
    //	int count=0;
    //    for (int i = n; i >0 ; i--) {
    //        bool flag=false;
    //        for (int j = 0; j < degsort[i].size(); ++j) {
    //            int k=degsort[i][j];
    //            if (trresult.find(k)!=trresult.end()){
    //				continue;
    //            }
    //            if(count>=99){
    //                flag=true;
    //                break;
    //            }
    //            count++;
    //        }
    //        if(flag){
    //            break;
    //        }
    //    }
    //    sdout.close();
    ofstream rout(graphname + "_result_new.txt", ios::out | ios::trunc | ios::binary);
    rout << "Tsize: " << Tsize << endl;
    rout << "K: " << K << endl;
    rout << "Method Name: " << method_name << endl;
    rout << "Time: " << timeuse / 1000000 << endl;
    rout << "Search Space:" << test_num << endl;
    rout << "===========================================" << endl;
    rout.close();
    printf("Method: %s. SumOfAns=%d. Time=%.1lfs. Search Space=%d\n", method_name, total_comp, timeuse / 1000000,
           test_num);
    printf("====================================================\n");
}

// Print the sum of Top-k values, Running Time(s), the number of vertices that have computed score.

// Print the Running Time(s) in File.
void FPrint_Time(char *str, double timeuse)
{
    FILE *fout;

    if (strlen(str) == 0)
    {
        printf("EMPTY Time Result Filename.\n");
        return;
    }
    fout = fopen(str, "a");
    fprintf(fout, "%ld\n", (long)(timeuse / 1000));
    fclose(fout);
}

void Print_Qnodes()
{
    ofstream sout("lj_qnodes_truss.txt", ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < qnodes.size(); ++i)
    {
        sout << qnodes[i] << " " << nodetruss[qnodes[i]] << endl;
    }
    sout.close();

    //    ofstream coreout("core_index.txt",ios::out|ios::trunc|ios::binary);
    //    for (int j = 0; j < n+1; ++j) {
    //        coreout<<j<<" "<<coreness[j]<<endl;
    //    }
    //    coreout.close();
}

void Print_ECCQnodes()
{
    ofstream sout("lj_query_ecc.txt", ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < qnodes.size(); ++i)
    {
        sout << 1 << " " << qnodes[i] << endl;
    }
    sout.close();

    //    ofstream coreout("core_index.txt",ios::out|ios::trunc|ios::binary);
    //    for (int j = 0; j < n+1; ++j) {
    //        coreout<<j<<" "<<coreness[j]<<endl;
    //    }
    //    coreout.close();
}

void Print_Communty(int v, double timeuse)
{
    string comfile = "lj_trusscom/" + to_string(v) + ".txt";
    string genfile = "lj_truss_query.csv";
    ofstream gout(genfile, ios::out | ios::trunc | ios::binary);
    gout << v << "," << nodetruss[v] << "," << comm.size() << "," << timeuse << endl;
    gout.close();
    ofstream sout(comfile, ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < comm.size(); ++i)
    {
        for (int j = 0; j < comm[i].size(); ++j)
        {
            int eid = comm[i][j];
            int x = eg[eid].x;
            int y = eg[eid].y;
            sout << x << " " << y << endl;
        }
        sout << "-1" << endl;
    }
    comm.clear();
    sout.close();
}

void Print_ktrussCommunty(char *maindir, char *output, int v, int k, double timeuse)
{
    string comfile = output + to_string(v) + ".txt";
    string mdir = maindir;
    string genfile = mdir + "truss_query.csv";
    ofstream gout(genfile, ios::out | ios::trunc | ios::binary);
    gout << v << "," << k << "," << comm.size() << "," << timeuse << endl;
    gout.close();
    ofstream sout(comfile, ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < comm.size(); ++i)
    {
        for (int j = 0; j < comm[i].size(); ++j)
        {
            int eid = comm[i][j];
            int x = eg[eid].x;
            int y = eg[eid].y;
            sout << x << " " << y << endl;
        }
        sout << "-1" << endl;
    }
    comm.clear();
    sout.close();
}

// void PrintkcoreCommunity(char* output,int v,int k)
// {
// 	// VKP: to create seperate folders each node
// 	string nodecomop=output+to_string(v)+"/";
// 	if(mkdir(nodecomop.c_str(),0777)==0)
//     {
//         cout<<"Output folder created:"<<nodecomop<<endl;
//     } else {
//         cout<<"output folder exists!:"<<nodecomop<<endl;
//     }

//     for (int i = 0; i < comm.size(); ++i) {
// 		string comfile=nodecomop+"kcore_k"+to_string(k)+"_com"+to_string(i+1)+".txt";
//     	ofstream sout(comfile,ios::out|ios::trunc|ios::binary);
//         for (int j = 0; j < comm[i].size(); ++j) {
//             int eid=comm[i][j];
//             int x=eg[eid].x;
//             int y=eg[eid].y;
//             sout<<x<<" "<<y<<endl;
//         }
//         sout<<"-1"<<endl;
// 		sout.close();
//     }
//     // comm.clear(); moved to main
// }

void PrintkcoreCommunity(char *output, int v, int k)
{
    // VKP: to create seperate folders each node
    string nodecomop = output + to_string(v) + "/";
    if (mkdir(nodecomop.c_str(), 0777) == 0)
    {
        cout << "Output folder created:" << nodecomop << endl;
    }
    else
    {
        cout << "output folder exists!:" << nodecomop << endl;
    }

    string comfile = nodecomop + "kcore_k" + to_string(k) + ".txt";
    ofstream sout(comfile, ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < comm.size(); ++i)
    {
        for (int j = 0; j < comm[i].size(); ++j)
        {
            int eid = comm[i][j];
            int x = eg[eid].x;
            int y = eg[eid].y;
            sout << x << " " << y << endl;
        }
        sout << "-1" << endl;
    }
    // comm.clear();
    sout.close();
    cout << "Print kcorecomm: Closed sout: " << v << endl;
}

void PrintkcoreCommunity_Nodes(char *output, int v, int k)
{
    string nodecomop = output + to_string(v) + "/";
    if (mkdir(nodecomop.c_str(), 0777) == 0)
        cout << "Output folder created:" << nodecomop << endl;
    else
        cout << "output folder exists!:" << nodecomop << endl;

    string comfile = nodecomop + "kcore_k" + to_string(k) + ".txt";
    ofstream sout(comfile, ios::out | ios::trunc | ios::binary);
    for (int i : kcore_visited_nodes)
        sout << i << "\n"; // don't flush the buffer repeatedly
    sout << "-1" << endl;
    // comm.clear();
    sout.close();
    cout << "Print kcorecomm: Closed sout: " << v << endl;
}

void Print_Core_Communty(int v, double timeuse)
{
    string comfile = "lj_corecom/" + to_string(communityID) + ".txt";
    string genfile = "lj_core_query.csv";
    ofstream gout(genfile, ios::out | ios::trunc | ios::binary);

    ofstream mout("lj_core_map.txt", ios::out | ios::trunc | ios::binary);
    mout << v << " " << communitymapping[v] << endl;
    mout.close();

    gout << v << "," << coreness[v] << "," << comm.size() << "," << timeuse << endl;
    gout.close();
    ofstream sout(comfile, ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < comm.size(); ++i)
    {
        for (int j = 0; j < comm[i].size(); ++j)
        {
            int eid = comm[i][j];
            int x = eg[eid].x;
            int y = eg[eid].y;
            sout << x << " " << y << endl;
        }
        sout << "-1" << endl;
    }
    comm.clear();
    sout.close();
}

void Print_Truss_Communty(int v, double timeuse)
{
    string comfile = "lj_trusscom/" + to_string(communityID) + ".txt";
    string genfile = "lj_truss_query.csv";
    ofstream gout(genfile, ios::out | ios::trunc | ios::binary);

    ofstream mout("lj_truss_map.txt", ios::out | ios::trunc | ios::binary);
    mout << v << " " << communitymapping[v] << endl;
    mout.close();

    gout << v << "," << nodetruss[v] << "," << comm.size() << "," << timeuse << endl;
    gout.close();
    ofstream sout(comfile, ios::out | ios::trunc | ios::binary);
    for (int i = 0; i < comm.size(); ++i)
    {
        for (int j = 0; j < comm[i].size(); ++j)
        {
            int eid = comm[i][j];
            int x = eg[eid].x;
            int y = eg[eid].y;
            sout << x << " " << y << endl;
        }
        sout << "-1" << endl;
    }
    comm.clear();
    sout.close();
}

int approximateDiameter(int iteration, map<int, vector<endpoint>> communinty, vector<int> nodelist)
{
    cout << "diameter" << endl;
    srand(time(NULL));
    int iter = iteration;
    int diameter = 0;
    if (nodelist.size() < iteration)
    {
        iter = nodelist.size();
    }

    unordered_set<int> randnode;
    int itc = 0;
    int nextnode;
    while (itc < iter)
    {
        int src;

        if (itc == 0)
        {
            int randpos = rand() % nodelist.size();
            src = nodelist[randpos];
        }
        else
        {
            src = nextnode;
        }
        itc++;
        //        if(randnode.find(randpos)!=randnode.end())
        //        {
        //            continue;
        //        }
        //        randnode.insert(randpos);

        //        cout<<randnode.size()<<endl;

        map<int, int> depthmap;
        queue<int> Q;
        unordered_set<int> visited_nodes;
        int local_max_depth;
        Q.push(src);
        visited_nodes.insert(src);
        depthmap[src] = 0;

        while (!Q.empty())
        {
            int v = Q.front();
            nextnode = v;
            Q.pop();
            int curDepth = depthmap[v];
            local_max_depth = curDepth;
            for (int i = 0; i < communinty[v].size(); i++)
            {
                int u = communinty[v][i].u;
                if (visited_nodes.find(u) == visited_nodes.end())
                {
                    visited_nodes.insert(u);
                    Q.push(u);
                    depthmap[u] = curDepth + 1;
                }
            }
        }
        if (diameter < local_max_depth)
        {
            diameter = local_max_depth;
        }

        cout << local_max_depth << endl;
    }

    return diameter;
}

double localCC(map<int, vector<endpoint>> community, vector<int> nodelist, map<pair<int, int>, int> edgeset)
{
    cout << "localcc begin" << endl;
    double *c = new double[nodelist.size()];
    double totalc = 0;
    for (int i = 0; i < nodelist.size(); i++)
    {
        int necount = 0;
        int v = nodelist[i];
        vector<endpoint> neighor_list = community[v];
        for (int j = 0; j < neighor_list.size() - 1; j++)
        {
            int x = neighor_list[j].u;
            for (int k = j + 1; k < neighor_list.size(); k++)
            {
                int y = neighor_list[k].u;
                if (edgeset.find(pair<int, int>(x, y)) != edgeset.end())
                {
                    necount++;
                }
            }
        }
        double ecount = necount;
        if (neighor_list.size() == 1)
        {
            c[i] = 0;
        }
        else
        {
            c[i] = 2 * ecount / (neighor_list.size() * (neighor_list.size() - 1));
        }

        totalc += c[i];
    }
    delete[] c;

    return totalc / nodelist.size();
}

double localCC_TL(map<int, vector<endpoint>> community, vector<int> nodelist, map<pair<int, int>, int> edgeset)
{
    cout << "localcc begin" << endl;
    double ecount;
    double total_cc;
    for (int i = 0; i < nodelist.size(); ++i)
    {
        int necount = 0;
        int v = nodelist[i];
        community[v];
        for (int j = 0; j < community[v].size(); ++j)
        {
            int u = community[v][j].u;

            int x = v;
            int y = u;
            if (community[v].size() > community[u].size())
            {
                x = u;
                y = v;
            }
            for (int k = 0; k < community[x].size(); ++k)
            {
                int z = community[x][k].u;
                if (edgeset.find(pair<int, int>(z, y)) != edgeset.end())
                {
                    necount++;
                }
            }
        }

        ecount = necount;
        double ci = ecount / (community[v].size() * (community[v].size() - 1));
        total_cc += ci;
    }

    return total_cc / nodelist.size();
}

void Read_Commnunities(const char *str, int querynode)
{
    ofstream sout("lj_core_com_eval.csv", ios::out | ios::trunc | ios::binary);
    FILE *fin = fopen(str, "r");
    int x, y;
    // bool isNewCom=true;
    map<int, vector<endpoint>> cl;
    map<pair<int, int>, int> cledgeset;
    unordered_set<int> nlist;
    vector<int> nl;
    int eid = 0;
    int clm = 0;
    int comcounter = 0;
    double total_avgdeg = 0;
    double total_avgden = 0;
    double total_avgcc = 0;
    double total_diameter = 0;
    double totoal_nd = 0;
    while (fscanf(fin, "%d", &x) != EOF)
    {
        if (x != -1)
        {
            fscanf(fin, "%d", &y);
            if (y < x)
                swap(x, y);
            if (x < y)
            {
                if (cledgeset.find(pair<int, int>(x, y)) == cledgeset.end())
                {
                    cledgeset[pair<int, int>(x, y)] = eid;
                    cledgeset[pair<int, int>(y, x)] = eid;

                    if (nlist.find(x) == nlist.end())
                    {
                        nlist.insert(x);
                        nl.push_back(x);
                    }
                    if (nlist.find(y) == nlist.end())
                    {
                        nlist.insert(y);
                        nl.push_back(y);
                    }

                    endpoint a, b;
                    a.u = y;
                    a.eid = eid;
                    cl[x].push_back(a);

                    b.u = x;
                    b.eid = eid;
                    cl[y].push_back(b);

                    eid++;
                    clm += 2;
                }
            }
        }
        else
        {
            cout << "Community: " << endl;
            double edgenum = clm / 2;
            cout << edgenum << endl;
            return;
            double nodenum = nl.size();

            double edge_density = edgenum * 2 / (nodenum * (nodenum - 1));
            int total_deg = 0;
            for (int i = 0; i < nl.size(); ++i)
            {
                int id = nl[i];
                total_deg += cl[id].size();
            }
            double avg_deg = total_deg / nodenum;
            cout << avg_deg << " " << edge_density << endl;
            double diameter = approximateDiameter(5, cl, nl);
            double local_cc = localCC(cl, nl, cledgeset);
            cout << diameter << " " << local_cc << endl;
            //
            comcounter++;
            totoal_nd += nodenum;

            //
            total_avgcc += local_cc;
            total_avgdeg += avg_deg;
            total_avgden += edge_density;
            total_diameter += diameter;

            cl.clear();
            cledgeset.clear();
            nl.clear();
            nlist.clear();
            eid = 0;
            clm = 0;
        }
    }

    sout << querynode << "," << total_diameter / comcounter << "," << total_avgdeg / comcounter << ","
         << total_avgden / comcounter << "," << total_avgcc / comcounter << "," << totoal_nd / comcounter << endl;
    //    sout<<totoal_nd/comcounter<<endl;
    sout.close();

    fclose(fin);
}

void CommunityEvaluation()
{
    //    char* infile="qnodes_core.txt";
    //    FILE *fin = fopen(infile, "r");
    //    int x,y;
    ////    for (int j = 0; j < 9; ++j) {
    ////        fscanf(fin, "%d %d", &x, &y);
    ////    }
    //
    //    while(fscanf(fin, "%d %d", &x, &y)!=EOF)
    //    {
    //        string comfile="corecom_bak/"+to_string(x)+".txt";
    //        cout<<"id: "<<x<<endl;
    //
    //        Read_Commnunities(comfile.c_str(),x);
    //    }
    //    fclose(fin);

    Read_Commnunities("lj_trusscom/1.txt", 1);
}
