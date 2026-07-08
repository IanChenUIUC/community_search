/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package experiments;

import cocktailParty.CocktailParty;
import coreGroups.CoreGroups;
import csm.CSM;
import csm.GraphWithOrderedAdj;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import main.Graph;

/**
 *
 * @author edotony
 */
public class ExperimentGraph {
    
    public static void statisticsOneNode(GraphWithOrderedAdj graph, int iterationsNumber, String datasetName) throws IOException {
        File mainDirectory = new File(datasetName + "OneNode");
        mainDirectory.mkdir();

        HashMap<Integer, Integer> cores = CoreGroups.coreGroupsAlgorithm(graph);

        QueryNodeByCore queryNode = new QueryNodeByCore(graph.getNodes(), cores);

        int numberOfEdges = graph.getNumberOfEdges();

        File dimensionsResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeDimensionsGraph.txt");
        FileWriter dimensionsFileWriter = new FileWriter(dimensionsResult);
        BufferedWriter dimensionsBufferedWriter = new BufferedWriter(dimensionsFileWriter);
        File executionTimesResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeExecutionTimesGraph.txt");
        FileWriter executionTimesFileWriter = new FileWriter(executionTimesResult);
        BufferedWriter executionTimesBufferedWriter = new BufferedWriter(executionTimesFileWriter);
        File densitiesResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeDensitiesGraph.txt");
        FileWriter densitiesFileWriter = new FileWriter(densitiesResult);
        BufferedWriter densitiesBufferedWriter = new BufferedWriter(densitiesFileWriter);

        ArrayList<Integer> samples = new ArrayList<Integer>();
        samples.add(1);
        samples.add(2);
        samples.add(4);
        samples.add(8);
        samples.add(16);
        samples.add(32);
        
        for (int i : samples) {
            File coreDirectory = new File(datasetName + "OneNode/Core" + i);
            coreDirectory.mkdir();
            System.out.println("\nCore: " + i);

            File map = new File(datasetName + "OneNode/Core" + i + "/map.txt");
            FileWriter fileWriter = new FileWriter(map);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

            int realIterationsNumber = iterationsNumber;
            if (i == 1) {
                realIterationsNumber = 100;
            }

            for (int j = 0; j < realIterationsNumber; j++) {
                ArrayList<Integer> queryNodes = queryNode.getQueryNodesOfCore(i);
                bufferedWriter.write(j + "\t" + queryNodes + "\n");

                Long[] executionTime = new Long[6];
                Integer[] dimension = new Integer[6];
                Double[] density = new Double[6];

                executionTime[0] = (long) i;
                dimension[0] = i;
                density[0] = (double) i;

                Graph coreGraph = CoreGroups.graphOfCore(graph, queryNodes, cores);
                dimension[1] = coreGraph.getNumberOfNodes();
                density[1] = ((double)coreGraph.getNumberOfEdges() / ((double)coreGraph.getNumberOfNodes() * (double)(coreGraph.getNumberOfNodes() - 1) / 2));

                if (j == 0) {
                    File coreGroupsFile = new File(datasetName + "OneNode/Core" + i + "/" + "core.txt");
                    FileWriter coreGroupsFileWriter = new FileWriter(coreGroupsFile);
                    BufferedWriter coreGroupsBufferedWriter = new BufferedWriter(coreGroupsFileWriter);
                    coreGroupsBufferedWriter.write(coreGraph.getNodes() + "");
                    coreGroupsBufferedWriter.close();
                    coreGroupsFileWriter.close();
                }

                long startTime = System.currentTimeMillis();
                HashSet<Integer> cocktailPartySet = CocktailParty.cocktailParty(graph, queryNodes);
                dimension[2] = cocktailPartySet.size();
                long endTime = System.currentTimeMillis();
                executionTime[2] = endTime - startTime;

                Graph cocktailPartyGraph = new Graph(graph, cocktailPartySet);
                if (coreGraph.getMinimumDegree() != cocktailPartyGraph.getMinimumDegree()) {
                    System.out.println("Crazy with node: " + queryNodes);
                }
                density[2] = ((double)cocktailPartyGraph.getNumberOfEdges() / ((double)cocktailPartyGraph.getNumberOfNodes() * (double)(cocktailPartyGraph.getNumberOfNodes() - 1) / 2));

                File cocktailPartyFile = new File(datasetName + "OneNode/Core" + i + "/" + j + "cocktailParty.txt");
                FileWriter cocktailPartyFileWriter = new FileWriter(cocktailPartyFile);
                BufferedWriter cocktailPartyBufferedWriter = new BufferedWriter(cocktailPartyFileWriter);
                cocktailPartyBufferedWriter.write(cocktailPartySet + "");
                cocktailPartyBufferedWriter.close();
                cocktailPartyFileWriter.close();

                startTime = System.currentTimeMillis();
                HashSet<Integer> CSMSet = CSM.CSMAlgorithm(graph, numberOfEdges, queryNodes.get(0), Integer.MIN_VALUE);
                dimension[3] = CSMSet.size();
                endTime = System.currentTimeMillis();
                executionTime[3] = endTime - startTime;

                Graph CSMGraph = new Graph(graph, CSMSet);
                density[3] = ((double)CSMGraph.getNumberOfEdges() / ((double)CSMGraph.getNumberOfNodes() * (double)(CSMGraph.getNumberOfNodes() - 1) / 2L));

                File csmGroupsFile = new File(datasetName + "OneNode/Core" + i + "/" + j + "csm.txt");
                FileWriter csmGroupsFileWriter = new FileWriter(csmGroupsFile);
                BufferedWriter csmGroupsBufferedWriter = new BufferedWriter(csmGroupsFileWriter);
                csmGroupsBufferedWriter.write(CSMSet + "");
                csmGroupsBufferedWriter.close();
                csmGroupsFileWriter.close();

                dimensionsBufferedWriter.append(dimension[0] + "\t" + dimension[1] + "\t" + dimension[2] + "\t" + dimension[3] + "\t" + dimension[4] + "\t" + dimension[5] + "\n");
                executionTimesBufferedWriter.write(executionTime[0] + "\t" + executionTime[1] + "\t" + executionTime[2] + "\t" + executionTime[3] + "\t" + executionTime[4] + "\t" + executionTime[5] + "\n");
                densitiesBufferedWriter.write(density[0] + "\t" + density[1] + "\t" + density[2] + "\t" + density[3] + "\t" + density[4] + "\t" + density[5] + "\n");

                dimensionsBufferedWriter.flush();
                executionTimesBufferedWriter.flush();
                densitiesBufferedWriter.flush();
            }

            bufferedWriter.close();
            fileWriter.close();
        }

        dimensionsBufferedWriter.close();
        executionTimesBufferedWriter.close();
        densitiesBufferedWriter.close();
        dimensionsFileWriter.close();
        executionTimesFileWriter.close();
        densitiesFileWriter.close();
    }
    
    public static void statisticsMoreNodes(Graph graph, int iterationsNumber, String datasetName) throws IOException {
        File mainDirectory = new File(datasetName + "MoreNodes");
        mainDirectory.mkdir();

        HashMap<Integer, Integer> cores = CoreGroups.coreGroupsAlgorithm(graph);

        File dimensionsResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesDimensionsGraph.txt");
        FileWriter dimensionsFileWriter = new FileWriter(dimensionsResult);
        BufferedWriter dimensionsBufferedWriter = new BufferedWriter(dimensionsFileWriter);
        File executionTimesResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesExecutionTimesGraph.txt");
        FileWriter executionTimesFileWriter = new FileWriter(executionTimesResult);
        BufferedWriter executionTimesBufferedWriter = new BufferedWriter(executionTimesFileWriter);
        File densitiesResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesDensitiesGraph.txt");
        FileWriter densitiesFileWriter = new FileWriter(densitiesResult);
        BufferedWriter densitiesBufferedWriter = new BufferedWriter(densitiesFileWriter);

        ArrayList<Integer> samples = new ArrayList<Integer>();
        samples.add(2);
        samples.add(4);
        samples.add(8);
        samples.add(16);
        samples.add(32);

        for (int i : samples) {
            File nodesDirectory = new File(datasetName + "MoreNodes/Nodes" + i);
            nodesDirectory.mkdir();
            System.out.println("\nQuery nodes: " + i);

            File map = new File(datasetName + "MoreNodes/Nodes" + i + "/map.txt");
            FileWriter fileWriter = new FileWriter(map);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);

            for (int j = 0; j < iterationsNumber; j++) {
                ArrayList<Integer> queryNodes = queryNodes(graph.getNodes(), i);
                bufferedWriter.write(j + "\t" + queryNodes + "\n");

                Long[] executionTime = new Long[6];
                Integer[] dimension = new Integer[6];
                Double[] density = new Double[6];

                executionTime[0] = (long) i;
                dimension[0] = i;
                density[0] = (double) i;

                Graph coreGraph = CoreGroups.graphOfCore(graph, queryNodes, cores);
                dimension[1] = coreGraph.getNumberOfNodes();
                density[1] = ((double)coreGraph.getNumberOfEdges() / ((double)coreGraph.getNumberOfNodes() * (double)(coreGraph.getNumberOfNodes() - 1) / 2));

                File coreGroupsFile = new File(datasetName + "MoreNodes/Nodes" + i + "/" + j + "core.txt");
                FileWriter coreGroupsFileWriter = new FileWriter(coreGroupsFile);
                BufferedWriter coreGroupsBufferedWriter = new BufferedWriter(coreGroupsFileWriter);
                coreGroupsBufferedWriter.write(coreGraph.getNodes() + "");
                coreGroupsBufferedWriter.close();
                coreGroupsFileWriter.close();

                long startTime = System.currentTimeMillis();
                HashSet<Integer> cocktailPartySet = CocktailParty.cocktailParty(graph, queryNodes);
                dimension[2] = cocktailPartySet.size();
                long endTime = System.currentTimeMillis();
                executionTime[2] = endTime - startTime;

                Graph cocktailPartyGraph = new Graph(graph, cocktailPartySet);
                if (coreGraph.getMinimumDegree() != cocktailPartyGraph.getMinimumDegree()) {
                    System.out.println("Crazy with nodes: " + queryNodes);
                }
                density[2] = ((double)cocktailPartyGraph.getNumberOfEdges() / ((double)cocktailPartyGraph.getNumberOfNodes() * (double)(cocktailPartyGraph.getNumberOfNodes() - 1) / 2));

                File cocktailPartyFile = new File(datasetName + "MoreNodes/Nodes" + i + "/" + j + "cocktailParty.txt");
                FileWriter cocktailPartyFileWriter = new FileWriter(cocktailPartyFile);
                BufferedWriter cocktailPartyBufferedWriter = new BufferedWriter(cocktailPartyFileWriter);
                cocktailPartyBufferedWriter.write(cocktailPartySet + "");
                cocktailPartyBufferedWriter.close();
                cocktailPartyFileWriter.close();

                dimensionsBufferedWriter.append(dimension[0] + "\t" + dimension[1] + "\t" + dimension[2] + "\t" + dimension[3] + "\t" + dimension[4] + "\t" + dimension[5] + "\n");
                executionTimesBufferedWriter.write(executionTime[0] + "\t" + executionTime[1] + "\t" + executionTime[2] + "\t" + executionTime[3] + "\t" + executionTime[4] + "\t" + executionTime[5] + "\n");
                densitiesBufferedWriter.write(density[0] + "\t" + density[1] + "\t" + density[2] + "\t" + density[3] + "\t" + density[4] + "\t" + density[5] + "\n");

                dimensionsBufferedWriter.flush();
                executionTimesBufferedWriter.flush();
                densitiesBufferedWriter.flush();
            }

            bufferedWriter.close();
            fileWriter.close();
        }

        dimensionsBufferedWriter.close();
        executionTimesBufferedWriter.close();
        densitiesBufferedWriter.close();
        dimensionsFileWriter.close();
        executionTimesFileWriter.close();
        densitiesFileWriter.close();
    }

    public static ArrayList<Integer> queryNodes(Set<Integer> largestConnectedComponent, int numberOfQueryNodes) {
        ArrayList<Integer> queryNodes = new ArrayList<Integer>();
        ArrayList<Integer> connectedComponentQueryNodesGenerator = new ArrayList<Integer>();
        connectedComponentQueryNodesGenerator.addAll(largestConnectedComponent);
        Random randomGenerator = new Random();

        int i = 0;
        while (i < numberOfQueryNodes) {
            int node = connectedComponentQueryNodesGenerator.get(randomGenerator.nextInt(connectedComponentQueryNodesGenerator.size()));
            if (!queryNodes.contains((int)node)) {
                queryNodes.add(node);
                i++;
            }
        }

        return queryNodes;
    }
}
