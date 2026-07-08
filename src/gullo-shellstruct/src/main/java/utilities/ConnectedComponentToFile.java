/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package utilities;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;

/**
 *
 * @author edotony
 */
public class ConnectedComponentToFile {
    
    public static void connectedComponentToFile(HashSet<Integer[]> edges, HashSet<Integer> connectedComponent, String datasetName) throws IOException {
        FileWriter fileWriter = new FileWriter(datasetName + "Reduced.txt");
        BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
        
        for (Integer[] edge : edges) {
            if ((connectedComponent.contains(edge[0])) && (connectedComponent.contains(edge[1]))) {
              bufferedWriter.write(edge[0] + "\t" + edge[1] + "\n");
            }
        }
        
        bufferedWriter.close();
        fileWriter.close();
    }
    
    
}
