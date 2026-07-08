/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package fromResults;

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
public class Diameter {
    
    public static void diameterFromResultsOneNode(Graph graph, String datasetName) throws IOException {
		
		File diameterResult = new File(datasetName + "OneNode/" + datasetName + "OneNodeDiameter.txt");
		FileWriter diameterFileWriter = new FileWriter(diameterResult);
		BufferedWriter diameterBufferedWriter = new BufferedWriter(diameterFileWriter);
		
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
                        double coreDiameter = 0;
			// For each query nodes set
			for (int id = 0; id < 12; id++) {
				Double[] diameter = new Double[6];
				
				diameter[0] = (double) i;
				
                                if (id == 0) {
                                    Graph coreGraph = new Graph(graph, core);
                                    coreDiameter = diameter(coreGraph);
                                }
				diameter[1] = coreDiameter;
				
				Graph cocktailPartyGraph = new Graph(graph, cocktailParty.get(id));
				diameter[2] = diameter(cocktailPartyGraph);
				
				Graph csmGraph = new Graph(graph, csm.get(id));
				diameter[3] = diameter(csmGraph);
				
				Graph rankGraph = new Graph(graph, rank.get(id));
				diameter[5] = diameter(rankGraph);
				
				diameterBufferedWriter.append(diameter[0] + "\t" + diameter[1] + "\t" + diameter[2] + "\t" + diameter[3] + "\t" + diameter[4] + "\t" + diameter[5] + "\n");
				diameterBufferedWriter.flush();
			}
		}
		
		diameterBufferedWriter.close();
		diameterFileWriter.close();
		
	}
	
	public static void diameterFromResultsMoreNodes(Graph graph, String datasetName) throws IOException {
		
		File diameterResult = new File(datasetName + "MoreNodes/" + datasetName + "MoreNodesDiameter.txt");
		FileWriter diameterFileWriter = new FileWriter(diameterResult);
		BufferedWriter diameterBufferedWriter = new BufferedWriter(diameterFileWriter);
		
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
			// For each query nodes set
			for (int id = 0; id < 12; id++) {
				Double[] diameter = new Double[6];
				
				diameter[0] = (double) i;
				
				Graph coreGraph = new Graph(graph, core.get(id));
				diameter[1] = diameter(coreGraph);
				
				Graph cocktailPartyGraph = new Graph(graph, cocktailParty.get(id));
				diameter[2] = diameter(cocktailPartyGraph);
				
				Graph rankSTGraph = new Graph(graph, rankST.get(id));
				diameter[4] = diameter(rankSTGraph);
				
				Graph rankGraph = new Graph(graph, rank.get(id));
				diameter[5] = diameter(rankGraph);
				
				diameterBufferedWriter.append(diameter[0] + "\t" + diameter[1] + "\t" + diameter[2] + "\t" + diameter[3] + "\t" + diameter[4] + "\t" + diameter[5] + "\n");
				diameterBufferedWriter.flush();
			}
		}
		
		diameterBufferedWriter.close();
		diameterFileWriter.close();
		
	}
	
	// Compute the diameter
	public static double diameter(Graph graph) {
		double diameter = -1;
				
		HashSet<Integer> nodes = new HashSet<Integer>(graph.getNodes());
		
		boolean inTime = true;
		long startTime = System.currentTimeMillis();
                int iterations = 0;
		while (inTime) {
			int node = nodes.iterator().next();
			DijkstraSP dijkstraSP = new DijkstraSP(graph, node);
			double f = dijkstraSP.maxDistTo();
			diameter = Math.max(diameter, f);
			nodes.remove(node);
                        iterations++;
			
			long endTime = System.currentTimeMillis();
			if ((endTime - startTime) > 720000 || nodes.isEmpty() || iterations >= 24) {
				inTime = false;
                        }
		}
				
		return diameter;
	}
}