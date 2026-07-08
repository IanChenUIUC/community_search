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

/**
 *
 * @author edotony
 */
public class Index {
    
    public static void indexFromResultsOneNode(int iterationsNumber, String path, String datasetName) throws IOException {
        IndexInterface index = new index.Index(path);
        
        File indexResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeIndexExecutionTime.txt");
        FileWriter indexFileWriter = new FileWriter(indexResult);
        BufferedWriter indexBufferedWriter = new BufferedWriter(indexFileWriter);
        
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
                Long[] indexTime = new Long[2];

                try {
                    indexTime[0] = (long) i;
                    long startTime = System.nanoTime();
                    index.getMinimumCoreIndex(queryNodes.get(j));
                    long endTime = System.nanoTime();
                    indexTime[1] = endTime - startTime;
                } catch (Exception e) {
                    
                }

                indexBufferedWriter.append(indexTime[0] + "\t" + indexTime[1] + "\n");
                indexBufferedWriter.flush();
            }
        }

        indexBufferedWriter.close();
        indexFileWriter.close();
    }

    public static void indexFromResultsMoreNodes(int iterationsNumber, String path, String datasetName) throws IOException {
        IndexInterface index = new index.Index(path);
        
        File indexResult = new File(datasetName + "MoreNodes/" + datasetName + "OneNodeIndexExecutionTime.txt");
        FileWriter indexFileWriter = new FileWriter(indexResult);
        BufferedWriter indexBufferedWriter = new BufferedWriter(indexFileWriter);
        
        ArrayList<Integer> samples = new ArrayList<Integer>();
        samples.add(2);
        samples.add(4);
        samples.add(8);
        samples.add(16);
        samples.add(32);

        for (int i : samples) {
            System.out.println("\nNodes: " + i);
            HashMap<Integer, ArrayList<Integer>> queryNodes = ResultsReader.queriesReader(datasetName + "MoreNodes/Nodes" + i);

            for (int j = 0; j < iterationsNumber; j++) {
                Long[] indexTime = new Long[2];

                try {
                    indexTime[0] = (long) i;
                    long startTime = System.nanoTime();
                    index.getMinimumCoreIndex(queryNodes.get(j));
                    long endTime = System.nanoTime();
                    indexTime[1] = endTime - startTime;
                } catch (Exception e) {
                    
                }

                indexBufferedWriter.append(indexTime[0] + "\t" + indexTime[1] + "\n");
                indexBufferedWriter.flush();
            }
        }

        indexBufferedWriter.close();
        indexFileWriter.close();
    }
}
