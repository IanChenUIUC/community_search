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
import rank.Connection;
import rank.Greedy;
import rank.GreedyConnection;

public class MainWSDM {

    public static void mainzzz(String[] args) throws IOException {
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

        GraphWithOrderedAdj graph = new GraphWithOrderedAdj("Datasets/" + network + ".txt");
        IndexInterface index = index(indexPath, indexKind, graph, network);

        ArrayList<Integer> queryNodes = query(query);
        queryNodesInGraph(graph, queryNodes);

        HashSet<Integer> solution = algorithm(algorithm, graph, index, queryNodes);
        output(solution, output);

        System.out.println("\nCommunity Search finished.");
    }

    private static void printUsage() {
        System.out.println("-n <network> -s <struct> -sk <struct_kind> -q <query> -a <algorithm> -o <output>");
    }

    private static HashSet<Integer> algorithm(String algorithm, GraphWithOrderedAdj graph, IndexInterface index, ArrayList<Integer> queryNodes) {
        HashSet<Integer> solution = null;
        if (algorithm.equals("gs")) {
            System.out.println("\nGlobal Search...");
            solution = CocktailParty.cocktailParty(graph, queryNodes);
        } else if (algorithm.equals("ls")) {
            if (queryNodes.size() > 1) {
                System.out.println("Local Search works only with one query node.");
                System.exit(0);
            }
            System.out.println("\nLocal Search...");
            solution = CSM.CSMAlgorithm(graph, graph.getNumberOfEdges(), queryNodes.get(0), Integer.MIN_VALUE);
        } else if (algorithm.equals("gr")) {
            System.out.println("\nGreedy...");
            solution = Greedy.heuristicRank(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        } else if (algorithm.equals("con")) {
            System.out.println("\nConnection...");
            solution = Connection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
        } else if (algorithm.equals("grcon")) {
            System.out.println("\nGreedyConnection...");
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

    private static void output(HashSet<Integer> solution, String output) throws IOException {
        System.out.println("\nWriting output...");
        FileWriter fileWriter = new FileWriter(output + ".txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

        bufferedWriter.write(solution + "");

        bufferedWriter.close();
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

}
