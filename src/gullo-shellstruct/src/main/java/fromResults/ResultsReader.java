/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package fromResults;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

/**
 *
 * @author edotony
 */
public class ResultsReader {
    
    public static HashMap<Integer, ArrayList<Integer>> queriesReader(String path) throws FileNotFoundException, IOException {
        HashMap<Integer, ArrayList<Integer>> queries = new HashMap<Integer, ArrayList<Integer>>();
    
        String line;
        String[] parts;
    
        FileReader fileReader = new FileReader(path);
        BufferedReader bufferedReader = new BufferedReader(fileReader);
    
        int index = 0;
        while ((line = bufferedReader.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) {
                continue;
            }
    
            ArrayList<Integer> queryNodes = new ArrayList<Integer>();
            parts = line.split(",");
            for (String string : parts) {
                queryNodes.add(Integer.parseInt(string.trim()));
            }
            queries.put(index, queryNodes);
    
            index++;
        }
    
        bufferedReader.close();
        fileReader.close();
    
        return queries;
    }
    
    public static HashMap<Integer, HashSet<Integer>> resultsReader(String path, String type) throws FileNotFoundException, IOException {
        HashMap<Integer, HashSet<Integer>> results = new HashMap<Integer, HashSet<Integer>>();
        
        String line;
	String[] parts;
        
        for (int i = 0; i < 30; i++) {
            FileReader fileReader = new FileReader(path + "/" + i + type);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            
            line = bufferedReader.readLine();
            line = line.replace("[", "");
            line = line.replace("]", "");
            
            parts = line.split(" ");
            
            results.put(i, new HashSet<Integer>());
            for (String string : parts) {
                results.get(i).add(Integer.parseInt(string.replace(",", "")));
            }
            
            bufferedReader.close();
            fileReader.close();
        }
        
        return results;
    }
    
    public static HashSet<Integer> coreReader(String path) throws FileNotFoundException, IOException {
        HashSet<Integer> result = new HashSet<Integer>();
        
        String line;
	String[] parts;
        
        FileReader fileReader = new FileReader(path + "/core.txt");
        BufferedReader bufferedReader = new BufferedReader(fileReader);
            
        line = bufferedReader.readLine();
        line = line.replace("[", "");
        line = line.replace("]", "");
            
        parts = line.split(" ");
            
        for (String string : parts) {
            result.add(Integer.parseInt(string.replace(",", "")));
        }
        
        bufferedReader.close();
        fileReader.close();
        
        return result;
    }
    
}
