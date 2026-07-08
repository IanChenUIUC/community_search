/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package dblp;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;

/**
 *
 * @author edotony
 */
public class DBLPCreator {

    public static void mappingFromFile() {
        System.out.println("Creating mapping...");
        HashSet<String> readNames = new HashSet<String>();
        try {
            FileReader fileReader = new FileReader("mapping.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            FileWriter fileWriter = new FileWriter("map.txt");
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            
            String[] parts;
            String line;
            
            while ((line = bufferedReader.readLine()) != null) {
                parts = line.split("\t");
                parts = parts[0].split("/");

                for (String part : parts) {
                    if (!readNames.contains(part)) {
                        bufferedWriter.write(readNames.size() + "\t" + part + "\n");
                        readNames.add(part);
                    }
                }
            }

            bufferedWriter.close();
            fileWriter.close();
            bufferedReader.close();
            fileReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found.");
            System.exit(0);
        } catch (IOException e) {
            System.out.println("Error in reading file.");
            System.exit(0);
        }

        System.out.println("Mapping created.");
    }

    public static void graphFromMapping(Mapping mapping) {
        System.out.println("Creating graph...");
        try {
            FileReader fileReader = new FileReader("mapping.txt");
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            FileWriter fileWriter = new FileWriter("dblp.txt");
            BufferedWriter bufferedWriter = new BufferedWriter(fileWriter);
            
            String line;
            String[] parts;
            
            while ((line = bufferedReader.readLine()) != null) {
                parts = line.split("\t");
                parts = parts[0].split("/");

                bufferedWriter.write(mapping.getIndex(parts[0]) + "\t" + mapping.getIndex(parts[1]) + "\n");
            }

            bufferedWriter.close();
            fileWriter.close();
            bufferedReader.close();
            fileReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("File not found.");
            System.exit(0);
        } catch (IOException e) {
            System.out.println("Error in reading file.");
            System.exit(0);
        }
        System.out.println("Graph created.");
    }
}