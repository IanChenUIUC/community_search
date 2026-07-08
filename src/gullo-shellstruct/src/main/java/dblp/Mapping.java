/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dblp;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 *
 * @author edotony
 */
public class Mapping {

    private final HashMap<String, Integer> nameIndex;
    private final HashMap<Integer, String> indexName;

    public Mapping() {
        this.nameIndex = new HashMap<String, Integer>();
        this.indexName = new HashMap<Integer, String>();

        System.out.println("Reading mapping...");
        try {
            FileReader fileReader = new FileReader("map.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            
            String line;
            String[] parts;
            
            while ((line = bufferedReader.readLine()) != null) {
                parts = line.split("\t");

                this.nameIndex.put(parts[1], Integer.valueOf(parts[0]));
                this.indexName.put(Integer.valueOf(parts[0]), parts[1]);
            }

            bufferedReader.close();
            fileReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found.");
            System.exit(0);
        } catch (IOException e) {
            System.out.println("Error in reading file.");
            System.exit(0);
        }

        System.out.println("Mapping read.");
    }

    public int getIndex(String name) {
        return this.nameIndex.get(name);
    }

    public String getName(int index) {
        return this.indexName.get(index);
    }

    public int getDimension() {
        return this.nameIndex.size();
    }
}
