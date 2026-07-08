
package main;

import csm.GraphWithOrderedAdj;
import index.TreeIndex;
import java.io.IOException;

public class BuildIndex_PKDDJ 
{
    public static void main(String[] args) throws IOException
    {
        String dataset = args[0];
        String output = args[1];
        
        GraphWithOrderedAdj graph = new GraphWithOrderedAdj(dataset);
        new TreeIndex(graph,dataset,output);
    }
}
