package index;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public interface IndexInterface {
	
	public int getMinimumCoreIndex(ArrayList<Integer> queryNodes);
	
	public HashSet<Integer> getNeighbors(int node, int coreIndex);
        
	public HashSet<Integer> getNeighbors(int node, int coreIndex, HashSet<Integer> subcore);
	
	public int getCoreMinimumDegree(int coreIndex);
	
	public int getNumberOfNodes(int coreIndex);

        public Set<Integer> getNodes();
        
        public HashSet<Integer> getCore(ArrayList<Integer> queryNodes);
}
