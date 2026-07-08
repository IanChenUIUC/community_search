
package experiments;

import coreGroups.CoreGroups;
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
import utilities.DFS;


public class SampleMultipleQueryNodes_VaryingCore 
{
    private static final Random randomGenerator = new Random();
    
    public static void main(String[] args) throws IOException
    {
        //String prefix = ".."+File.separator;
        String prefix = "";
        String pathDataset = prefix+"Datasets"+File.separator;
        String pathOutput = prefix+"Experiments"+File.separator;
        
        //String[] datasets = new String[]{"jazzReduced.txt","celegansneuralReduced.txt","Email-EnronReduced.txt","web-NotreDameReduced.txt","web-GoogleNiceReduced.txt","youtube-linksNiceReduced.txt","as-skitterNiceReduced.txt","web-BerkStanReduced.txt","wikipedia-20051105NiceReduced.txt","flickrReduced.txt"};
        //String[] outputs = new String[]{"jazz","celegansneural","Email-Enron","web-NotreDame","web-GoogleNice","youtube-linksNice","as-skitterNice","web-BerkStan","wikipedia-20051105Nice","flickr"};
        //String[] datasets = new String[]{"web-GoogleNiceReduced.txt","youtube-linksNiceReduced.txt","as-skitterNiceReduced.txt","web-BerkStanReduced.txt","wikipedia-20051105NiceReduced.txt","flickrReduced.txt"};
        //String[] outputs = new String[]{"web-GoogleNice","youtube-linksNice","as-skitterNice","web-BerkStan","wikipedia-20051105Nice","flickr"};
        String[] datasets = new String[]{"web-NotreDameReduced.txt"};
        String[] outputs = new String[]{"web-NotreDame"};
        
        
        int[] qnNumbers = new int[]{2,4,8,16,32};
        int[] cNumbers = new int[]{1,2,4,8,16,32};
        int samples = 30;
        int samplesPerCore = samples/cNumbers.length;
        
        for(int h=0; h<datasets.length; h++)
        {
            String d = datasets[h];
            System.out.println("\n\nDataset "+d);
            String[][][] sampled = new String[cNumbers.length][qnNumbers.length][samplesPerCore];
            String dpath = pathDataset+d;
            GraphWithOrderedAdj graph = new GraphWithOrderedAdj(dpath); //load dataset
            HashMap<Integer, Integer> cores = CoreGroups.coreGroupsAlgorithm(graph); //compute core decomposition
            for(int i=0; i<cNumbers.length; i++)
            {
                int c = cNumbers[i];
                System.out.println("Core "+c);
                HashSet<Integer> coreNodes = getCoreNodes(cores,c); //get core c
                ArrayList<Integer> giantConnectedComponent = getCoreGiantConnectedComponent(graph,coreNodes);
                
                for(int j=0; j<qnNumbers.length; j++)
                {
                    int qn = qnNumbers[j];
                    if(giantConnectedComponent.size() >= qn)
                    {
                        for(int s=0; s<samplesPerCore; s++)
                        {
                            String sample = sample(giantConnectedComponent,qn);
                            sampled[i][j][s] = sample;
                        }
                    }
                }
            }
            
            //write output
            for(int j=0; j<qnNumbers.length; j++)
            {
                int qn =  qnNumbers[j];
                String dirPath = pathOutput+outputs[h]+"MoreNodes"+File.separator+"Nodes"+qn+"_byCore";
                new File(dirPath).mkdirs();
                
                BufferedWriter bw = new BufferedWriter(new FileWriter(dirPath+File.separator+"map.txt"));
                int id = 0;
                for(int s=0; s<sampled[0][0].length; s++)
                {
                    for(int i=0; i<cNumbers.length; i++)
                    {
                        String sample = sampled[i][j][s];
                        if(sample != null)
                        {
                            bw.write(""+id+"\t"+sample);
                            bw.newLine();
                            id++;
                        }
                    }
                }
                bw.flush();
                bw.close();
            }
        }
    }

    private static HashSet<Integer> getCoreNodes(HashMap<Integer, Integer> cores, int c) 
    {
        HashSet<Integer> core = new HashSet<Integer>();
        for(int x:cores.keySet())
        {
            if(cores.get(x)>=c)
            {
                core.add(x);
            }
        }
        
        return core;
    }

    private static ArrayList<Integer> getCoreGiantConnectedComponent(GraphWithOrderedAdj graph, HashSet<Integer> coreNodes)
    {
        Graph coreSubgraph = new Graph(graph, coreNodes);
        int mindeg = coreSubgraph.getMinimumDegree();
        //int n1 = graph.getNumberOfNodes();
        //int m1 = graph.getNumberOfEdges();
        //int n2 = coreSubgraph.getNumberOfNodes();
        //int m2 = coreSubgraph.getNumberOfEdges();
        HashSet<Integer> giantCC = DFS.largestConnectedComponent(coreSubgraph);
        ArrayList<Integer> cc = new ArrayList<Integer>();
        for(int x:giantCC)
        {
            cc.add(x);
        }
        
        return cc;
    }

    private static String sample(ArrayList<Integer> giantConnectedComponent, int qn)
    {
        Set<Integer> sampled = new HashSet<Integer>();
        while(sampled.size()<qn)
        {
            int rnd = randomGenerator.nextInt(giantConnectedComponent.size());
            int x = giantConnectedComponent.get(rnd);
            if(!sampled.contains(x))
            {
                sampled.add(x);
            }
        }
        
        ArrayList<Integer> v = new ArrayList<Integer>();
        v.addAll(sampled);
        
            return v.toString();
    }
}
