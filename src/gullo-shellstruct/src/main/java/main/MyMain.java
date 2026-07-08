package main;

import cocktailParty.CocktailParty;
import csm.CSM;
import csm.GraphWithOrderedAdj;
import fromResults.ResultsReader;
import index.Index;
import index.IndexInterface;
import index.TreeIndex;
import rank.Connection;
import rank.Greedy;
import rank.GreedyConnection;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

// Usage:
//   -n <network file> -s <index path> -sk cs|ss
//   -q <directory containing map.txt, in the "n1,n2,n3 per line" format>
//   -a <dash-separated algorithms, e.g. gr-con-grcon-gs-ls-core>
//   -o <output directory>
//
// For every query and every algorithm, writes the returned node set to:
//   <output dir>/<algorithm>/<queryIndex>.txt
// as a single comma-separated line, e.g. "42,7,100023".
public class MyMain {

    public static void main(String[] args) throws IOException {
        System.out.println("Community Search (write communities) started...\n");

        if (args.length == 0) {
            printUsage();
            return;
        }

        String network = null;
        String indexPath = null;
        String indexKind = null;
        String queryDir = null;
        String[] algorithms = null;
        String outputDir = null;

        for (int i = 0; i < args.length; i++) {
            if (args[i].equalsIgnoreCase("--help")) {
                printUsage();
                return;
            }
            if (args[i].equals("-n")) { network = args[i + 1]; i++; }
            if (args[i].equals("-s")) { indexPath = args[i + 1]; i++; }
            if (args[i].equals("-sk")) { indexKind = args[i + 1]; i++; }
            if (args[i].equals("-q")) { queryDir = args[i + 1]; i++; }
            if (args[i].equals("-a")) { algorithms = args[i + 1].split("-"); i++; }
            if (args[i].equals("-o")) { outputDir = args[i + 1]; i++; }
        }

        GraphWithOrderedAdj graph = new GraphWithOrderedAdj(network);
        IndexInterface index = index(indexPath, indexKind, graph, network);

        HashMap<Integer, ArrayList<Integer>> queries = ResultsReader.queriesReader(queryDir);

        for (String algorithm : algorithms) {
            File algorithmDir = new File(outputDir + File.separator + algorithm);
            algorithmDir.mkdirs();

            System.out.println("\nAlgorithm: " + algorithm);
            for (int queryIndex : queries.keySet()) {
                ArrayList<Integer> queryNodes = queries.get(queryIndex);

                long start = System.currentTimeMillis();
                HashSet<Integer> solution = runAlgorithm(algorithm, graph, index, queryNodes);
                long elapsed = System.currentTimeMillis() - start;

                writeCommunity(algorithmDir.getPath() + File.separator + queryIndex + ".txt", solution);
                System.out.println("  query " + queryIndex + ": size=" + solution.size() + ", time=" + elapsed + "ms");
            }
        }

        System.out.println("\nDone.");
    }

    private static void printUsage() {
        System.out.println("-n <network> -s <index> -sk cs|ss -q <query dir with map.txt> -a <alg-alg-alg> -o <output dir>");
    }

    private static HashSet<Integer> runAlgorithm(String algorithm, GraphWithOrderedAdj graph, IndexInterface index, ArrayList<Integer> queryNodes) {
        switch (algorithm) {
            case "gs":
                return CocktailParty.cocktailParty(graph, queryNodes);
            case "ls":
                if (queryNodes.size() > 1) {
                    System.out.println("Local Search (ls) only supports a single query node; skipping.");
                    return new HashSet<Integer>();
                }
                return CSM.CSMAlgorithm(graph, graph.getNumberOfEdges(), queryNodes.get(0), Integer.MIN_VALUE);
            case "gr":
                return Greedy.heuristicRank(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
            case "con":
                return Connection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
            case "grcon":
                return GreedyConnection.heuristicRankWithST(index, queryNodes, index.getMinimumCoreIndex(queryNodes));
            case "core":
                return index.getCore(queryNodes);
            default:
                System.out.println("Unknown algorithm: " + algorithm);
                System.exit(0);
                return null; // unreachable
        }
    }

    private static void writeCommunity(String path, HashSet<Integer> solution) throws IOException {
        BufferedWriter bw = new BufferedWriter(new FileWriter(path));
        StringBuilder sb = new StringBuilder();
        for (int node : solution) {
            if (sb.length() > 0) sb.append(",");
            sb.append(node);
        }
        bw.write(sb.toString());
        bw.newLine();
        bw.close();
    }

    private static IndexInterface index(String indexPath, String indexKind, Graph graph, String network) {
        if (indexKind.equals("cs")) {
            return new Index(indexPath);
        } else if (indexKind.equals("ss")) {
            return new TreeIndex(indexPath);
        } else {
            System.out.println("This kind of struct doesn't exist.");
            System.exit(0);
            return null; // unreachable
        }
    }
}
