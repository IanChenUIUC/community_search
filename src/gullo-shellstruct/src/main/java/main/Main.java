package main;

import fromResults.Diameter;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException {
        
        Graph graph = new Graph(args[0]);
        //Diameter.diameterFromResultsMoreNodes(graph, args[1]);
        Diameter.diameterFromResultsOneNode(graph, args[1]);
        
    }
}