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
import rank.Connection;
import rank.Greedy;
import rank.GreedyConnection;

public class MainPKDDJ {

    public static void main(String[] args) throws IOException {
        /*
        //-n $n -s $s -sk ss -q $q -a $a -o $o      
        String n = "Datasets"+File.separator+"Email-EnronReduced.txt";
        String s = n+"ShellStruct";
        String sk = "ss";
        String qn = "32";
        String qq = "Experiments"+File.separator+"Email-EnronMoreNodes"+File.separator+"Nodes"+qn;
        String a = "grcon";
        String o = "Results"+File.separator+"EmailEnron_"+a+"_"+qn+".txt"; 
        args = new String[]{"-n",n,"-s",s,"-sk",sk,"-q",qq,"-a",a,"-o",o};
        */
        
        System.out.println("Community Search started...\n");

        if (args.length == 0) {
            printUsage();
            return;
        }

        String network = null;
        String indexPath = null;
        String indexKind = null;
        String query = null;
        String algorithm = null;
        String output = null;

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
            if (args[i].equals("-q")) {
                query = args[i + 1];
                i++;
            }
            if (args[i].equals("-a")) {
                algorithm = args[i + 1];
                i++;
            }
            if (args[i].equals("-o")) {
                output = args[i + 1];
                i++;
            }
        }

        GraphWithOrderedAdj graph = new GraphWithOrderedAdj(network);
        IndexInterface index = index(indexPath, indexKind, graph, network);

        HashMap<Integer, ArrayList<Integer>> queryNodes = queryFromFile(query);
        //queryNodesInGraph(graph, queryNodes);

        double[] sizes = new double[queryNodes.size()];
        double[] densities = new double[queryNodes.size()];
        double[] times = new double[queryNodes.size()];
        for(int i=0; i<queryNodes.size(); i++)
        {
            System.out.println("\nRun "+(i+1)+" of "+queryNodes.size());
            ArrayList<Integer> q = queryNodes.get(i);
            
            long start = System.currentTimeMillis();
            HashSet<Integer> solution = algorithm(algorithm, graph, index, q);
            
            times[i] = System.currentTimeMillis()-start;
            sizes[i] = solution.size();
            
            Graph solutionSubgraph = new Graph(graph, solution);
            densities[i] = ((double)solutionSubgraph.getNumberOfEdges() / ((double)solutionSubgraph.getNumberOfNodes() * (double)(solutionSubgraph.getNumberOfNodes() - 1) / 2L));
        }
        
        double[] avgMedianSize = getAvgMedian(sizes);
        double[] avgMedianDensity = getAvgMedian(densities);
        double[] avgMedianTime = getAvgMedian(times);
        
        output(output,network,indexKind,algorithm,queryNodes.size(),avgMedianSize,avgMedianDensity,avgMedianTime);

        System.out.println("\nCommunity Search finished.");
    }

    private static void printUsage() {
        System.out.println("-n <network> -s <struct> -sk <struct_kind> -q <query> -a <algorithm> -o <output>");
    }

    private static HashSet<Integer> algorithm(String algorithm, GraphWithOrderedAdj graph, IndexInterface index, ArrayList<Integer> queryNodes) {
        HashSet<Integer> solution = null;
        if (algorithm.equals("gs")) {
            System.out.println("Global Search...");
            solution = CocktailParty.cocktailParty(graph, queryNodes);
        } else if (algorithm.equals("ls")) {
            if (queryNodes.size() > 1) {
                System.out.println("Local Search works only with one query node.");
                System.exit(0);
            }
            System.out.println("Local Search...");
            solution = CSM.CSMAlgorithm(graph, graph.getNumberOfEdges(), queryNodes.get(0), Integer.MIN_VALUE);
        } else if (algorithm.equals("gr")) {
            System.out.println("Greedy...");
            solution = Greedy.heuristicRank(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        } else if (algorithm.equals("con")) {
            System.out.println("Connection...");
            solution = Connection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        } else if (algorithm.equals("grcon")) {
            System.out.println("GreedyConnection...");
            solution = GreedyConnection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
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

    private static void output(String output, String network, String indexKind, String algorithm, int runs, double[] avgMedianSize, double[] avgMedianDensity, double[] avgMedianTime) throws IOException {
        System.out.println("\nWriting output...");
        FileWriter fileWriter = new FileWriter(output + ".txt");
        BufferedWriter bw = new BufferedWriter(fileWriter);
        
        bw.write("Dataset:\t"+network); bw.newLine();
        bw.write("Index type:\t"+indexKind); bw.newLine();
        bw.write("Algorithm:\t"+algorithm); bw.newLine();
        bw.write("Runs:\t"+runs); bw.newLine();
        bw.newLine();
        bw.write("Results (avg, median):"); bw.newLine();
        bw.write("Size\t"+avgMedianSize[0]+"\t"+avgMedianSize[1]); bw.newLine();
        bw.write("Density\t"+avgMedianDensity[0]+"\t"+avgMedianDensity[1]); bw.newLine();
        bw.write("Time\t"+avgMedianTime[0]+"\t"+avgMedianTime[1]); bw.newLine();
        
        bw.flush();
        bw.close();
        fileWriter.close();
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

    private static double[] getAvgMedian(double[] sizes) 
    {
        double sum = 0.0;
        for(double d:sizes)
        {
            sum  += d;
        }
        sum /= sizes.length;
        
        Arrays.sort(sizes);
        
        return new double[]{sum,sizes[sizes.length/2]};
    }

}
