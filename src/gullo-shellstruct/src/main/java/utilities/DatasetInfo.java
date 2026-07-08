/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package utilities;

import java.io.IOException;
import java.util.HashSet;
import main.Graph;

/**
 *
 * @author edotony
 */
public class DatasetInfo {
    
    public static void datasetInfo(Graph graph, String datasetName) throws IOException {
        HashSet<Integer> largestConnectedComponent = DFS.largestConnectedComponent(graph);

        System.out.println("\nThe density of the graph is: " + (double)(graph.getNumberOfEdges() / (double)(graph.getNumberOfNodes() * (graph.getNumberOfNodes() - 1) / 2)));

        System.out.println("\nThe real number of undirected nodes is: " + graph.getNumberOfEdges());
        System.out.println("\nThe average degree of the graph is: " + avgDegree(graph));
        
        ConnectedComponentToFile.connectedComponentToFile(graph.getEdges(), largestConnectedComponent, datasetName);
    }
    
    private static double avgDegree(Graph graph) {
        double degreeSum = 0.0;
        
        for (int node : graph.getNodes()) {
            degreeSum += graph.getDegree(node);
        }

        return degreeSum / graph.getNumberOfNodes();
    }
}
