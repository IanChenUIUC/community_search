package main;

import index.Index;
import index.IndexInterface;
import index.TreeIndex;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;

import cocktailParty.CocktailParty;
import csm.CSM;
import csm.GraphWithOrderedAdj;
import fromResults.ResultsReader;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.StringTokenizer;
import rank.Connection;
import rank.Greedy;
import rank.GreedyConnection;

public class MainPKDDJ_MultipleParameters_ConstrainedCocktailParty
{

    public static void main(String[] args) throws IOException
    {
            
        //String n = "Datasets"+File.separator+"Email-EnronReduced.txt";
        String n = "Datasets"+File.separator+"web-NotreDameReduced.txt";
        String s = n+"ShellStruct";
        String sk = "ss";
        //String qp = "Experiments"+File.separator+"Email-EnronMoreNodes"+File.separator+"Nodes";
        //String qp = "Experiments"+File.separator+"Email-EnronOneNode"+File.separator+"Core";
        String qp = "Experiments"+File.separator+"web-NotreDameOneNode"+File.separator+"Core";
        //String qn = "2-4-8-16-32";
        String qn = "1-2-4-8-16-32";
        String a = "gr-gskfast";
        //String o = "Results"+File.separator+"EmailEnron_ConstrainedCP"; 
        String o = "Results"+File.separator+"WebNotreDame_ConstrainedCP";
        //args = new String[]{"-n",n,"-s",s,"-sk",sk,"-qp",qp,"-qn",qn,"-a",a,"-o",o};
        args = new String[]{"-n",n,"-s",s,"-sk",sk,"-qp",qp,"-qn",qn,"-a",a,"-o",o,"-singlenode"};
        
        
        System.out.println("Community Search started...\n");

        if (args.length == 0) {
            printUsage();
            return;
        }
        
        
        int maxRuns = 30;
        boolean appendRnd = true;
        String network = null;
        String indexPath = null;
        String indexKind = null;
        String queryPath = null;
        String[] queryNodesNumber = null;
        String[] algorithms = null;
        String output = null;
        boolean singlenode = false;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equalsIgnoreCase("--help")) {
                printUsage();
                return;
            }
            if (args[i].equals("-n")) {
                network = args[i + 1];
                i++;
            }
            if (args[i].equals("-s")) {
                indexPath = args[i + 1];
                i++;
            }
            if (args[i].equals("-sk")) {
                indexKind = args[i + 1];
                i++;
            }
            if (args[i].equals("-qp")) {
                queryPath = args[i + 1];
                i++;
            }
            if (args[i].equals("-qn")) {
                queryNodesNumber = splitParameter(args[i + 1]);
                i++;
            }
            if (args[i].equals("-a")) {
                algorithms = splitParameter(args[i + 1]);
                i++;
            }
            if (args[i].equals("-o")) {
                output = args[i + 1];
                i++;
            }
            if (args[i].equals("-singlenode")) {
                singlenode = true;
                i++;
            }
        }
        
        GraphWithOrderedAdj graph = new GraphWithOrderedAdj(network);
        IndexInterface index = index(indexPath, indexKind, graph, network);
        if(graph.getNumberOfEdges() >= 10000000)
        {
            maxRuns = 10;
        }
        String rnd = (appendRnd)?("_"+System.currentTimeMillis()):"";
        String sn = (singlenode)?"_SingleNode":"";
        BufferedWriter bwsum = new BufferedWriter(new FileWriter(output+"_SUMMARY"+sn+rnd+".txt"));
        BufferedWriter bwdet = new BufferedWriter(new FileWriter(output+"_DETAILS"+sn+rnd+".txt"));
        printHeader(bwsum,network,graph.getNumberOfNodes(), graph.getNumberOfEdges(),indexKind,Math.min(maxRuns, 30));
        printHeader(bwdet,network,graph.getNumberOfNodes(), graph.getNumberOfEdges(),indexKind,Math.min(maxRuns, 30));
        
        for (String aaa:queryNodesNumber)
        {
            System.out.println("\n\n\n----------------------------------------------------------------------------------------");
            if(singlenode)
            {
                System.out.println("Running experiments for core "+aaa);
            }
            else
            {
                System.out.println("Running experiments for "+aaa+" query nodes");
            }
            
            HashMap<Integer, ArrayList<Integer>> queryNodes = queryFromFile(queryPath+aaa);
            //queryNodesInGraph(graph, queryNodes);
            int qns = Math.min(maxRuns, queryNodes.size());
           
            double[][] sizes = new double[qns+2][algorithms.length];
            double[][] densities = new double[qns+2][algorithms.length];
            double[][] times = new double[qns+2][algorithms.length];
            double[][] mindegrees = new double[qns+2][algorithms.length];

            int[] kkk = new int[qns];
            for(int j=0; j<algorithms.length; j++)
            {
                String bbb = algorithms[j];
                System.out.print("\nMethod: "+bbb);
                if(bbb.equals("gskfast"))
                {
                    System.out.print(" (size constraints: ");
                    for(int z:kkk)
                    {
                        System.out.print(" "+z);
                    }
                    System.out.print(")");
                }
                System.out.println("");
                System.out.println("Total runs: "+qns);
                System.out.print("Run ");

                for(int i=0; i<qns; i++)
                {
                    System.out.print(""+(i+1)+" ");
                    ArrayList<Integer> q = queryNodes.get(i);
                    
                    /*
                    if(aaa.equals("4") && bbb.equals("gskfast") && i==1)
                    {
                        int forDebug = 0;
                    }
                    */

                    long start = System.currentTimeMillis();
                    HashSet<Integer> solution = algorithm(bbb, graph, index, q, kkk[i]);

                    times[i][j] = System.currentTimeMillis()-start;
                    sizes[i][j] = solution.size();
                    if(j==0)
                    {
                        kkk[i] = (int)sizes[i][j];
                    }
                    

                    
                    Graph solutionSubgraph = new Graph(graph, solution);
                    densities[i][j] = ((double)solutionSubgraph.getNumberOfEdges() / ((double)solutionSubgraph.getNumberOfNodes() * (double)(solutionSubgraph.getNumberOfNodes() - 1) / 2L));
                    mindegrees[i][j] = solutionSubgraph.computeMinimumDegree();
                    
                }
                System.out.println("");


                double[] avgMedianSize = getAvgMedian(sizes,j);
                double[] avgMedianDensity = getAvgMedian(densities,j);
                double[] avgMedianTime = getAvgMedian(times,j);
                double[] avgMedianMinDegree = getAvgMedian(mindegrees,j);
                
                for(int k=0; k<avgMedianSize.length; k++)
                {
                    sizes[sizes.length-2+k][j] = avgMedianSize[k];
                    densities[densities.length-2+k][j] = avgMedianDensity[k];
                    times[times.length-2+k][j] = avgMedianTime[k];
                    mindegrees[mindegrees.length-2+k][j] = avgMedianMinDegree[k];
                }

                //output(output,network,indexKind,algorithms,queryNodes.size(),avgMedianSize,avgMedianDensity,avgMedianTime);
            }
            System.out.println("----------------------------------------------------------------------------------------");
            printOutputSummary(bwsum,aaa,algorithms,sizes,densities,times,mindegrees,singlenode);
            printOutputDetails(bwdet,aaa,algorithms,sizes,densities,times,mindegrees,singlenode);
        }
        bwsum.flush(); bwsum.close();
        bwdet.flush(); bwdet.close();
        System.out.println("\nCommunity Search finished.");
    }

    private static void printUsage() {
        System.out.println("-n <network> -s <struct> -sk <struct_kind> -q <query> -a <algorithm> -o <output>");
    }

    private static HashSet<Integer> algorithm(String algorithm, GraphWithOrderedAdj graph, IndexInterface index, ArrayList<Integer> queryNodes, int k) {
        HashSet<Integer> solution = null;
        if (algorithm.equals("gs")) {
            //System.out.println("Global Search...");
            solution = CocktailParty.cocktailParty(graph, queryNodes);
        } else if (algorithm.equals("ls")) {
            if (queryNodes.size() > 1) {
                System.out.println("Local Search works only with one query node.");
                System.exit(0);
            }
            //System.out.println("Local Search...");
            solution = CSM.CSMAlgorithm(graph, graph.getNumberOfEdges(), queryNodes.get(0), Integer.MIN_VALUE);
        } else if (algorithm.equals("gr")) {
            //System.out.println("Greedy...");
            solution = Greedy.heuristicRank(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        } else if (algorithm.equals("con")) {
            //System.out.println("Connection...");
            solution = Connection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        } else if (algorithm.equals("grcon")) {
            //System.out.println("GreedyConnection...");
            solution = GreedyConnection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        }  else if (algorithm.equals("core")) {
            //System.out.println("GreedyConnection...");
            solution = index.getCore(queryNodes);
        }  else if (algorithm.equals("gskfast")) {
            //System.out.println("GreedyConnection...");
            solution = CocktailParty.cocktailParty_constrained_GreedyFast(graph, queryNodes, k);
        } else {
            System.out.println("Wrong algorithm selection.");
            System.exit(0);
        }

        return solution;
    }

    private static ArrayList<Integer> query(String query) {
        ArrayList<Integer> queryNodes = new ArrayList<Integer>();
        String[] parts = query.split("/");

        for (String part : parts) {
            queryNodes.add(Integer.parseInt(part));
        }

        System.out.println("\nQuery nodes: " + queryNodes);
        return queryNodes;
    }
    
    private static HashMap<Integer, ArrayList<Integer>> queryFromFile(String queryFilePath) throws IOException{

        return ResultsReader.queriesReader(queryFilePath);
    }

    private static IndexInterface index(String indexPath, String indexKind, Graph graph, String network) {
        IndexInterface index = null;
        if (indexPath == null) {
            System.out.println("Creating struct...");
            if (indexKind.equals("cs")) {
                index = new Index(graph, network);
            } else if (indexKind.equals("ss")) {
                index = new TreeIndex(graph, network);
            } else {
                System.out.println("This kind of struct doesn't exist.");
                System.exit(0);
            }
            System.out.println("Struct created.");
        } else {
            System.out.println("Reading struct...");
            if (indexKind.equals("cs")) {
                index = new Index(indexPath);
            } else if (indexKind.equals("ss")) {
                index = new TreeIndex(indexPath);
            } else {
                System.out.println("This kind of struct doesn't exist.");
                System.exit(0);
            }
            System.out.println("Struct read.");
        }

        return index;
    }

    private static void queryNodesInGraph(Graph graph, ArrayList<Integer> queryNodes) {
        for (int queryNode : queryNodes) {
            if (!graph.doesNodeExist(queryNode)) {
                System.out.println("The node " + queryNode + " is not in the network.");
                System.exit(0);
            }
        }
    }

    private static double[] getAvgMedian(double[][] sizes, int j) 
    {
        double[] v = new double[sizes.length-2];
        double sum = 0.0;
        for(int i=0; i<sizes.length-2; i++)
        {
            sum  += sizes[i][j];
            v[i] = sizes[i][j];
        }
        sum /= sizes.length-2;
        
        Arrays.sort(v);
        
        return new double[]{sum,v[v.length/2]};
    }

    private static String[] splitParameter(String s) 
    {
        ArrayList<String> tmp = new ArrayList<String>();
        StringTokenizer st = new StringTokenizer(s,"-/");
        while(st.hasMoreTokens())
        {
            tmp.add(st.nextToken());
        }
        String[] v = new String[tmp.size()];
        tmp.toArray(v);
        
        return v;
    }

    private static void printHeader(BufferedWriter bw, String network, int nodes, int edges, String indexKind, int runs) throws IOException
    {
        bw.write("Dataset:\t"+network); bw.newLine();
        bw.write("#nodes:\t"+nodes); bw.newLine();
        bw.write("#edges:\t"+edges); bw.newLine();
        bw.write("Index type:\t"+indexKind); bw.newLine();
        bw.write("Runs:\t"+runs); bw.newLine();
        bw.newLine();
        bw.newLine();
        bw.newLine();
        bw.flush();
    }

    private static void printOutputSummary(BufferedWriter bw, String qn, String[] algorithms, double[][] sizes, double[][] densities, double[][] times, double[][] mindegrees, boolean singlenode) throws IOException
    {
        if(!singlenode)
        {
            bw.write("#query nodes="+qn);
        }
        else
        {
            bw.write("core "+qn);
        }
        for(String a:algorithms)
        {
            bw.write("\t"+a);
        }
        bw.newLine();
        
        bw.write("median size");
        for(double x:sizes[sizes.length-1])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        
        bw.write("median time");
        for(double x:times[times.length-1])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        
        bw.write("median density");
        for(double x:densities[densities.length-1])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        
        bw.write("median mindeg");
        for(double x:mindegrees[mindegrees.length-1])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        
        bw.write("avg size");
        for(double x:sizes[sizes.length-2])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        
        bw.write("avg time");
        for(double x:times[times.length-2])
        {
            bw.write("\t"+x);
        }
        bw.newLine();

        bw.write("avg density");
        for(double x:densities[densities.length-2])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        
        bw.write("avg mindeg");
        for(double x:mindegrees[mindegrees.length-2])
        {
            bw.write("\t"+x);
        }
        bw.newLine();
        bw.newLine();
        bw.newLine();
        bw.flush();
    }

    private static void printOutputDetails(BufferedWriter bw, String qn, String[] algorithms, double[][] sizes, double[][] densities, double[][] times, double[][] mindegrees, boolean singlenode) throws IOException
    {
        if(!singlenode)
        {
            bw.write("#query nodes="+qn);
        }
        else
        {
            bw.write("core "+qn);
        }
        for(String a:algorithms)
        {
            bw.write("\t"+a);
        }
        bw.newLine();
        
        for(int i=0; i<sizes.length-2; i++)
        {
            bw.write("size (run "+i+")");
            for(int j=0; j<sizes[i].length; j++)
            {
                bw.write("\t"+sizes[i][j]);
            }
            bw.newLine();
        }
        bw.write("--------------------------------------------------------------------------");
        bw.newLine();
        
        for(int i=0; i<times.length-2; i++)
        {
            bw.write("time (run "+i+")");
            for(int j=0; j<times[i].length; j++)
            {
                bw.write("\t"+times[i][j]);
            }
            bw.newLine();
        }
        bw.write("--------------------------------------------------------------------------");
        bw.newLine();

        for(int i=0; i<densities.length-2; i++)
        {
            bw.write("density (run "+i+")");
            for(int j=0; j<densities[i].length; j++)
            {
                bw.write("\t"+densities[i][j]);
            }
            bw.newLine();
        }
        bw.write("--------------------------------------------------------------------------");
        bw.newLine();
        
        for(int i=0; i<mindegrees.length-2; i++)
        {
            bw.write("mindeg (run "+i+")");
            for(int j=0; j<mindegrees[i].length; j++)
            {
                bw.write("\t"+mindegrees[i][j]);
            }
            bw.newLine();
        }
        bw.write("--------------------------------------------------------------------------");
        bw.newLine();
        bw.newLine();
        bw.newLine();
        bw.flush();
    }

}
