/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package experiments;

import fromResults.ResultsReader;
import index.IndexInterface;
import index.TreeIndex;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import main.Graph;
import rank.Connection;
import rank.Greedy;

/**
 *
 * @author edotony
 */
public class ExperimentIndex {

    public static void statisticsOneNode(int iterationsNumber, String path, String datasetName) throws IOException {
        IndexInterface index = new TreeIndex(path);

        File dimensionsResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeDimensionsIndex.txt");
        FileWriter dimensionsFileWriter = new FileWriter(dimensionsResult);
        BufferedWriter dimensionsBufferedWriter = new BufferedWriter(dimensionsFileWriter);
        File executionTimesResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeExecutionTimesIndex.txt");
        FileWriter executionTimesFileWriter = new FileWriter(executionTimesResult);
        BufferedWriter executionTimesBufferedWriter = new BufferedWriter(executionTimesFileWriter);
        File densitiesResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeDensitiesIndex.txt");
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
            System.out.println("\nCore: " + i);
            HashMap<Integer, ArrayList<Integer>> queryNodes = ResultsReader.queriesReader(datasetName + "OneNode/Core" + i);

            int realIterationsNumber = iterationsNumber;
            if (i == 1) {
                realIterationsNumber = 100;
            }

            for (int j = 0; j < realIterationsNumber; j++) {
                Long[] executionTime = new Long[6];
                Integer[] dimension = new Integer[6];
                Double[] density = new Double[6];

                executionTime[0] = (long) i;
                dimension[0] = i;
                density[0] = (double) i;

                HashSet<Integer> rankSet;
                try {
                    long startTime = System.currentTimeMillis();
                    int minimumCoreIndex = index.getMinimumCoreIndex(queryNodes.get(j));
                    long endTime = System.currentTimeMillis();
                    executionTime[4] = endTime - startTime;

                    startTime = System.currentTimeMillis();
                    rankSet = Greedy.heuristicRank(index, queryNodes.get(j), minimumCoreIndex);
                    dimension[5] = rankSet.size();
                    endTime = System.currentTimeMillis();
                    executionTime[5] = endTime - startTime;

                    Graph rankGraph = new Graph(index, rankSet);
                    density[5] = ((double) rankGraph.getNumberOfEdges() / ((double) rankGraph.getNumberOfNodes() * (double) (rankGraph.getNumberOfNodes() - 1) / 2));

                } catch (Exception e) {
                    rankSet = new HashSet<Integer>();
                }

                File rankGroupsFile = new File(datasetName + "OneNode/Core" + i + "/" + j + "rank.txt");
                FileWriter rankGroupsFileWriter = new FileWriter(rankGroupsFile);
                BufferedWriter rankGroupsBufferedWriter = new BufferedWriter(rankGroupsFileWriter);
                rankGroupsBufferedWriter.write(rankSet + "");
                rankGroupsBufferedWriter.close();
                rankGroupsFileWriter.close();

                dimensionsBufferedWriter.append(dimension[0] + "\t" + dimension[1] + "\t" + dimension[2] + "\t" + dimension[3] + "\t" + dimension[4] + "\t" + dimension[5] + "\n");
                executionTimesBufferedWriter.write(executionTime[0] + "\t" + executionTime[1] + "\t" + executionTime[2] + "\t" + executionTime[3] + "\t" + executionTime[4] + "\t" + executionTime[5] + "\n");
                densitiesBufferedWriter.write(density[0] + "\t" + density[1] + "\t" + density[2] + "\t" + density[3] + "\t" + density[4] + "\t" + density[5] + "\n");

                dimensionsBufferedWriter.flush();
                executionTimesBufferedWriter.flush();
                densitiesBufferedWriter.flush();
            }
        }

        dimensionsBufferedWriter.close();
        executionTimesBufferedWriter.close();
        densitiesBufferedWriter.close();
        dimensionsFileWriter.close();
        executionTimesFileWriter.close();
        densitiesFileWriter.close();
    }

    public static void statisticsMoreNodes(int iterationsNumber, String path, String datasetName) throws IOException {
        IndexInterface index = new TreeIndex(path);

        File dimensionsResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesDimensionsIndex.txt");
        FileWriter dimensionsFileWriter = new FileWriter(dimensionsResult);
        BufferedWriter dimensionsBufferedWriter = new BufferedWriter(dimensionsFileWriter);
        File executionTimesResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesExecutionTimesIndex.txt");
        FileWriter executionTimesFileWriter = new FileWriter(executionTimesResult);
        BufferedWriter executionTimesBufferedWriter = new BufferedWriter(executionTimesFileWriter);
        File densitiesResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesDensitiesIndex.txt");
        FileWriter densitiesFileWriter = new FileWriter(densitiesResult);
        BufferedWriter densitiesBufferedWriter = new BufferedWriter(densitiesFileWriter);

        ArrayList<Integer> samples = new ArrayList<Integer>();
        samples.add(2);
        samples.add(4);
        samples.add(8);
        samples.add(16);
        samples.add(32);

        for (int i : samples) {
            System.out.println("\nCore: " + i);
            HashMap<Integer, ArrayList<Integer>> queryNodes = ResultsReader.queriesReader(datasetName + "MoreNodes/Nodes" + i);

            for (int j = 0; j < iterationsNumber; j++) {
                Long[] executionTime = new Long[6];
                Integer[] dimension = new Integer[6];
                Double[] density = new Double[6];

                executionTime[0] = (long) i;
                dimension[0] = i;
                density[0] = (double) i;

                HashSet<Integer> rankSTSet;
                HashSet<Integer> rankConnSet;
                try {
                    long startTime = System.currentTimeMillis();
                    int minimumCoreIndex = index.getMinimumCoreIndex(queryNodes.get(j));
                    long endTime = System.currentTimeMillis();
                    executionTime[3] = endTime - startTime;

                    startTime = System.currentTimeMillis();
                    rankSTSet = Connection.heuristicRankWithST(index, queryNodes.get(j), minimumCoreIndex);
                    dimension[4] = rankSTSet.size();
                    endTime = System.currentTimeMillis();
                    executionTime[4] = endTime - startTime;

                    Graph rankSTGraph = new Graph(index, rankSTSet);
                    density[4] = ((double) rankSTGraph.getNumberOfEdges() / ((double) rankSTGraph.getNumberOfNodes() * (double) (rankSTGraph.getNumberOfNodes() - 1) / 2));

                    startTime = System.currentTimeMillis();
                    rankConnSet = Greedy.heuristicRank(index, queryNodes.get(j), minimumCoreIndex);
                    dimension[5] = rankConnSet.size();
                    endTime = System.currentTimeMillis();
                    executionTime[5] = endTime - startTime;

                    Graph rankConnGraph = new Graph(index, rankConnSet);
                    density[5] = ((double) rankConnGraph.getNumberOfEdges() / ((double) rankConnGraph.getNumberOfNodes() * (double) (rankConnGraph.getNumberOfNodes() - 1) / 2));

                } catch (Exception e) {
                    rankSTSet = new HashSet<Integer>();
                    rankConnSet = new HashSet<Integer>();
                }

                File rankSTGroupsFile = new File(datasetName + "MoreNodes/Nodes" + i + "/" + j + "rankST.txt");
                FileWriter rankSTGroupsFileWriter = new FileWriter(rankSTGroupsFile);
                BufferedWriter rankSTGroupsBufferedWriter = new BufferedWriter(rankSTGroupsFileWriter);
                rankSTGroupsBufferedWriter.write(rankSTSet + "");
                rankSTGroupsBufferedWriter.close();
                rankSTGroupsFileWriter.close();

                File rankConnGroupsFile = new File(datasetName + "MoreNodes/Nodes" + i + "/" + j + "rank.txt");
                FileWriter rankConnGroupsFileWriter = new FileWriter(rankConnGroupsFile);
                BufferedWriter rankConnGroupsBufferedWriter = new BufferedWriter(rankConnGroupsFileWriter);
                rankConnGroupsBufferedWriter.write(rankConnSet + "");
                rankConnGroupsBufferedWriter.close();
                rankConnGroupsFileWriter.close();

                dimensionsBufferedWriter.append(dimension[0] + "\t" + dimension[1] + "\t" + dimension[2] + "\t" + dimension[3] + "\t" + dimension[4] + "\t" + dimension[5] + "\n");
                executionTimesBufferedWriter.write(executionTime[0] + "\t" + executionTime[1] + "\t" + executionTime[2] + "\t" + executionTime[3] + "\t" + executionTime[4] + "\t" + executionTime[5] + "\n");
                densitiesBufferedWriter.write(density[0] + "\t" + density[1] + "\t" + density[2] + "\t" + density[3] + "\t" + density[4] + "\t" + density[5] + "\n");

                dimensionsBufferedWriter.flush();
                executionTimesBufferedWriter.flush();
                densitiesBufferedWriter.flush();
            }
        }

        dimensionsBufferedWriter.close();
        executionTimesBufferedWriter.close();
        densitiesBufferedWriter.close();
        dimensionsFileWriter.close();
        executionTimesFileWriter.close();
        densitiesFileWriter.close();
    }
    
}
