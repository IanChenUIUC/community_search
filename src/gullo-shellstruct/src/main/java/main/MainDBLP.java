package main;

import cocktailParty.CocktailParty;
import csm.CSM;
import csm.GraphWithOrderedAdj;
import dblp.Mapping;
import index.IndexInterface;
import index.TreeIndex;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import rank.Connection;
import rank.Greedy;

public class MainDBLP {

    public static void mainzzz(String[] args) throws IOException {
        System.out.println("Community Search started...\n");
        
        if (args.length == 0) {
            printUsage();
            return;
        }

        String query = null;
        String algorithm = null;
        String output = null;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equalsIgnoreCase("--help")) {
                printUsage();
                return;
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

        GraphWithOrderedAdj graph = new GraphWithOrderedAdj("dblpReduced.txt");
        IndexInterface index = new TreeIndex("dblpShellStruct");
        Mapping mapping = new Mapping();

        ArrayList<Integer> queryNodes = query(query, mapping);
        queryNodesInGraph(graph, queryNodes);

        HashSet<Integer> solution = algorithm(algorithm, graph, index, queryNodes);
        output(solution, output, mapping);
                
        System.out.println("\nCommunity Search finished.");
    }

    private static void printUsage() {
        System.out.println("-q <query> -a <algorithm> -o <output>");
    }

    private static HashSet<Integer> algorithm(String algorithm, GraphWithOrderedAdj graph, IndexInterface index, ArrayList<Integer> queryNodes) {
        HashSet<Integer> solution = null;
        switch (algorithm) {
            case "gs":
                System.out.println("\nGlobal Search...");
                solution = CocktailParty.cocktailParty(graph, queryNodes);
                break;
            case "ls":
                if (queryNodes.size() > 1) {
                    System.out.println("Local Search works only with one query node.");
                    System.exit(0);
                }
                System.out.println("\nLocal Search...");
                solution = CSM.CSMAlgorithm(graph, graph.getNumberOfEdges(), queryNodes.get(0), Integer.MIN_VALUE);
                break;
            case "gr":
                System.out.println("\nGreedy...");
                solution = Greedy.heuristicRank(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
                break;
            case "con":
                System.out.println("\nConnection...");
                solution = Connection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
                break;
            default:
                System.out.println("Wrong algorithm selection.");
                System.exit(0);
        }

        return solution;
    }

    private static ArrayList<Integer> query(String query, Mapping mapping) {
        ArrayList<Integer> queryNodes = new ArrayList<Integer>();
        String[] parts = query.split("/");

        for (String part : parts) {
            queryNodes.add(mapping.getIndex(part));
        }

        System.out.println("\nQuery nodes: " + queryNodes);
        return queryNodes;
    }

    private static void output(HashSet<Integer> solution, String output, Mapping mapping) throws IOException {
        System.out.println("\nWriting output...");
        FileWriter fileWriter = new FileWriter(output + ".txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        for (Integer node : solution) {
            bufferedWriter.write(mapping.getName(node) + "\n");
        }

        bufferedWriter.close();
        fileWriter.close();
    }

    private static void queryNodesInGraph(Graph graph, ArrayList<Integer> queryNodes) {
        for (int queryNode : queryNodes) {
            if (!graph.doesNodeExist(queryNode)) {
                System.out.println("The node " + queryNode + " is not in the network.");
                System.exit(0);
            }
        }
    }

}
