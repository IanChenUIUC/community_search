/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package experiments;

import static experiments.ExperimentGraph.queryNodes;
import index.IndexInterface;
import index.TreeIndex;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import main.Graph;
import rank.Connection;
import rank.ConnectionLimited;
import rank.Greedy;

/**
 *
 * @author edotony
 */
public class ExperimentNew {
    
    public static void statisticsMoreNodes(Graph graph, int iterationsNumber, String path, String datasetName) throws IOException {
        IndexInterface index = new TreeIndex(path);

        File mainDirectory = new File(datasetName + "MoreNodesHeuristics");
        mainDirectory.mkdir();
        
        File dimensionsResult = new File(datasetName + "MoreNodesHeuristics/" + datasetName + "MoreNodesDimensions.txt");
        FileWriter dimensionsFileWriter = new FileWriter(dimensionsResult);
        BufferedWriter dimensionsBufferedWriter = new BufferedWriter(dimensionsFileWriter);
        File executionTimesResult = new File(datasetName + "MoreNodesHeuristics/" + datasetName + "MoreNodesExecutionTimes.txt");
        FileWriter executionTimesFileWriter = new FileWriter(executionTimesResult);
        BufferedWriter executionTimesBufferedWriter = new BufferedWriter(executionTimesFileWriter);
        File densitiesResult = new File(datasetName + "MoreNodesHeuristics/" + datasetName + "MoreNodesDensities.txt");
        FileWriter densitiesFileWriter = new FileWriter(densitiesResult);
        BufferedWriter densitiesBufferedWriter = new BufferedWriter(densitiesFileWriter);

        ArrayList<Integer> samples = new ArrayList<Integer>();
        samples.add(2);
        samples.add(4);
        samples.add(8);
        samples.add(16);
        samples.add(32);

        for (int i : samples) {
            File nodesDirectory = new File(datasetName + "MoreNodesHeuristics/Nodes" + i);
            nodesDirectory.mkdir();
            System.out.println("\nNodes: " + i);
            
            File map = new File(datasetName + "MoreNodesHeuristics/Nodes" + i + "/map.txt");
            FileWriter fileWriter = new FileWriter(map);
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            
            for (int j = 0; j < iterationsNumber; j++) {
                ArrayList<Integer> queryNodes = queryNodes(graph.getNodes(), i);
                bufferedWriter.write(j + "\t" + queryNodes + "\n");
                
                Long[] executionTime = new Long[8];
                Integer[] dimension = new Integer[8];
                Double[] density = new Double[8];

                executionTime[0] = (long) i;
                dimension[0] = i;
                density[0] = (double) i;

                HashSet<Integer> rankSTSet;
                HashSet<Integer> rankConnSet;
                HashSet<Integer> rankConnLim3Set;
                HashSet<Integer> rankConnLim4Set;
                try {
                    int minimumCoreIndex = index.getMinimumCoreIndex(queryNodes);

                    long startTime = System.currentTimeMillis();
                    rankSTSet = Connection.heuristicRankWithST(index, queryNodes, minimumCoreIndex);
                    dimension[4] = rankSTSet.size();
                    long endTime = System.currentTimeMillis();
                    executionTime[4] = endTime - startTime;

                    Graph rankSTGraph = new Graph(index, rankSTSet);
                    density[4] = ((double) rankSTGraph.getNumberOfEdges() / ((double) rankSTGraph.getNumberOfNodes() * (double) (rankSTGraph.getNumberOfNodes() - 1) / 2));

                    startTime = System.currentTimeMillis();
                    rankConnSet = Greedy.heuristicRank(index, queryNodes, minimumCoreIndex);
                    dimension[5] = rankConnSet.size();
                    endTime = System.currentTimeMillis();
                    executionTime[5] = endTime - startTime;

                    Graph rankConnGraph = new Graph(index, rankConnSet);
                    density[5] = ((double) rankConnGraph.getNumberOfEdges() / ((double) rankConnGraph.getNumberOfNodes() * (double) (rankConnGraph.getNumberOfNodes() - 1) / 2));

                    startTime = System.currentTimeMillis();
                    rankConnLim3Set = ConnectionLimited.heuristicRankWithST(index, queryNodes, minimumCoreIndex, 3);
                    dimension[6] = rankConnLim3Set.size();
                    endTime = System.currentTimeMillis();
                    executionTime[6] = endTime - startTime;
                    
                    Graph rankConnLim3Graph = new Graph(index, rankConnLim3Set);
                    density[6] = ((double) rankConnLim3Graph.getNumberOfEdges() / ((double) rankConnLim3Graph.getNumberOfNodes() * (double) (rankConnLim3Graph.getNumberOfNodes() - 1) / 2));
                    
                    startTime = System.currentTimeMillis();
                    rankConnLim4Set = ConnectionLimited.heuristicRankWithST(index, queryNodes, minimumCoreIndex, 4);
                    dimension[7] = rankConnLim4Set.size();
                    endTime = System.currentTimeMillis();
                    executionTime[7] = endTime - startTime;
                    
                    Graph rankConnLim4Graph = new Graph(index, rankConnLim4Set);
                    density[7] = ((double) rankConnLim4Graph.getNumberOfEdges() / ((double) rankConnLim4Graph.getNumberOfNodes() * (double) (rankConnLim4Graph.getNumberOfNodes() - 1) / 2));
                    
                } catch (Exception e) {
                    rankSTSet = new HashSet<Integer>();
                    rankConnSet = new HashSet<Integer>();
                }

                File rankSTGroupsFile = new File(datasetName + "MoreNodesHeuristics/Nodes" + i + "/" + j + "rankST.txt");
                FileWriter rankSTGroupsFileWriter = new FileWriter(rankSTGroupsFile);
                BufferedWriter rankSTGroupsBufferedWriter = new BufferedWriter(rankSTGroupsFileWriter);
                rankSTGroupsBufferedWriter.write(rankSTSet + "");
                rankSTGroupsBufferedWriter.close();
                rankSTGroupsFileWriter.close();

                File rankConnGroupsFile = new File(datasetName + "MoreNodesHeuristics/Nodes" + i + "/" + j + "rank.txt");
                FileWriter rankConnGroupsFileWriter = new FileWriter(rankConnGroupsFile);
                BufferedWriter rankConnGroupsBufferedWriter = new BufferedWriter(rankConnGroupsFileWriter);
                rankConnGroupsBufferedWriter.write(rankConnSet + "");
                rankConnGroupsBufferedWriter.close();
                rankConnGroupsFileWriter.close();
                
                File rankConnLim3GroupsFile = new File(datasetName + "MoreNodesHeuristics/Nodes" + i + "/" + j + "rankLim3.txt");
                FileWriter rankConnLim3GroupsFileWriter = new FileWriter(rankConnLim3GroupsFile);
                BufferedWriter rankConnLim3GroupsBufferedWriter = new BufferedWriter(rankConnLim3GroupsFileWriter);
                rankConnLim3GroupsBufferedWriter.write(rankConnSet + "");
                rankConnLim3GroupsBufferedWriter.close();
                rankConnLim3GroupsFileWriter.close();
                
                File rankConnLim4GroupsFile = new File(datasetName + "MoreNodesHeuristics/Nodes" + i + "/" + j + "rankLim4.txt");
                FileWriter rankConnLim4GroupsFileWriter = new FileWriter(rankConnLim4GroupsFile);
                BufferedWriter rankConnLim4GroupsBufferedWriter = new BufferedWriter(rankConnLim4GroupsFileWriter);
                rankConnLim4GroupsBufferedWriter.write(rankConnSet + "");
                rankConnLim4GroupsBufferedWriter.close();
                rankConnLim4GroupsFileWriter.close();

                dimensionsBufferedWriter.append(dimension[0] + "\t" + dimension[1] + "\t" + dimension[2] + "\t" + dimension[3] + "\t" + dimension[4] + "\t" + dimension[5] + "\t" + dimension[6] + "\t" + dimension[7] + "\n");
                executionTimesBufferedWriter.write(executionTime[0] + "\t" + executionTime[1] + "\t" + executionTime[2] + "\t" + executionTime[3] + "\t" + executionTime[4] + "\t" + executionTime[5] + "\n" + executionTime[6] + "\t" + executionTime[7] + "\n");
                densitiesBufferedWriter.write(density[0] + "\t" + density[1] + "\t" + density[2] + "\t" + density[3] + "\t" + density[4] + "\t" + density[5] + "\t" + density[6] + "\t" + density[7] + "\n");

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
    
}
