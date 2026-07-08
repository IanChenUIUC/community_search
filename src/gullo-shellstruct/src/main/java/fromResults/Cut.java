/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fromResults;

import index.IndexInterface;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import main.Graph;

/**
 *
 * @author edotony
 */
public class Cut {

    public static void cutFromResultsOneNode(Graph graph, String datasetName) throws IOException {
        File cutResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeCut.txt");
        FileWriter cutFileWriter = new FileWriter(cutResult);
        BufferedWriter cutBufferedWriter = new BufferedWriter(cutFileWriter);

        ArrayList<Integer> cores = new ArrayList<Integer>();
        cores.add(1);
        cores.add(2);
        cores.add(4);
        cores.add(8);
        cores.add(16);
        cores.add(32);

        for (int i : cores) {
            HashSet<Integer> core = ResultsReader.coreReader(datasetName + "OneNode/Core" + i);
            HashMap<Integer, HashSet<Integer>> cocktailParty = ResultsReader.resultsReader(datasetName + "OneNode/Core" + i, "cocktailParty.txt");
            HashMap<Integer, HashSet<Integer>> csm = ResultsReader.resultsReader(datasetName + "OneNode/Core" + i, "csm.txt");
            HashMap<Integer, HashSet<Integer>> rank = ResultsReader.resultsReader(datasetName + "OneNode/Core" + i, "rank.txt");

            System.out.println("\nCore: " + i);

            int realIterationsNumber = 30;
            if (i == 1) {
                realIterationsNumber = 100;
            }

            for (int j = 0; j < realIterationsNumber; j++) {
                Integer[] cut = new Integer[6];

                cut[0] = i;

                Graph coreGraph = new Graph(graph, core);
                cut[1] = cut(coreGraph, graph);

                Graph cocktailPartyGraph = new Graph(graph, (HashSet) cocktailParty.get(Integer.valueOf(j)));
                cut[2] = cut(cocktailPartyGraph, graph);

                Graph csmGraph = new Graph(graph, (HashSet) csm.get(Integer.valueOf(j)));
                cut[3] = cut(csmGraph, graph);

                Graph rankGraph = new Graph(graph, (HashSet) rank.get(Integer.valueOf(j)));
                cut[5] = cut(rankGraph, graph);

                cutBufferedWriter.append(cut[0] + "\t" + cut[1] + "\t" + cut[2] + "\t" + cut[3] + "\t" + cut[4] + "\t" + cut[5] + "\n");
                cutBufferedWriter.flush();
            }

            cutBufferedWriter.close();
            cutFileWriter.close();
        }
    }

    

    public static void cutFromResultsMoreNodes(Graph graph, String datasetName) throws IOException {
        File cutResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesCut.txt");
        FileWriter cutFileWriter = new FileWriter(cutResult);
        BufferedWriter cutBufferedWriter = new BufferedWriter(cutFileWriter);

        ArrayList<Integer> cores = new ArrayList<Integer>();
        cores.add(2);
        cores.add(4);
        cores.add(8);
        cores.add(16);
        cores.add(32);

        for (int i : cores) {
            HashMap<Integer, HashSet<Integer>> core = ResultsReader.resultsReader(datasetName + "MoreNodes/Nodes" + i, "core.txt");
            HashMap<Integer, HashSet<Integer>> cocktailParty = ResultsReader.resultsReader(datasetName + "MoreNodes/Nodes" + i, "cocktailParty.txt");
            HashMap<Integer, HashSet<Integer>> rankST = ResultsReader.resultsReader(datasetName + "MoreNodes/Nodes" + i, "rankST.txt");
            HashMap<Integer, HashSet<Integer>> rank = ResultsReader.resultsReader(datasetName + "MoreNodes/Nodes" + i, "rank.txt");

            System.out.println("\nNodes: " + i);

            for (int j = 0; j < 30; j++) {
                Integer[] cut = new Integer[6];

                cut[0] = i;

                Graph coreGraph = new Graph(graph, (HashSet) core.get(Integer.valueOf(j)));
                cut[1] = cut(coreGraph, graph);

                Graph cocktailPartyGraph = new Graph(graph, (HashSet) cocktailParty.get(Integer.valueOf(j)));
                cut[2] = cut(cocktailPartyGraph, graph);

                Graph rankSTGraph = new Graph(graph, (HashSet) rankST.get(Integer.valueOf(j)));
                cut[4] = cut(rankSTGraph, graph);

                Graph rankGraph = new Graph(graph, (HashSet) rank.get(Integer.valueOf(j)));
                cut[5] = cut(rankGraph, graph);

                cutBufferedWriter.append(cut[0] + "\t" + cut[1] + "\t" + cut[2] + "\t" + cut[3] + "\t" + cut[4] + "\t" + cut[5] + "\n");
                cutBufferedWriter.flush();
            }

            cutBufferedWriter.close();
            cutFileWriter.close();
        }
    }

    

    public static int cut(Graph cuttedGraph, Graph graph) {
        int cut = 0;

        for (int node : cuttedGraph.getNodes()) {
            HashSet<Integer> neighbors = new HashSet(graph.getNeighbors(node));
            neighbors.removeAll(cuttedGraph.getNeighbors(node));
            cut += neighbors.size();
        }

        return cut;
    }

    public static int cut(Graph cuttedGraph, IndexInterface index) {
        int cut = 0;

         for (int node : cuttedGraph.getNodes()) {
             HashSet<Integer> neighbors = new HashSet(index.getNeighbors(node, 0));
            neighbors.removeAll(cuttedGraph.getNeighbors(node));
            cut += neighbors.size();
        }

        return cut;
    }
}
