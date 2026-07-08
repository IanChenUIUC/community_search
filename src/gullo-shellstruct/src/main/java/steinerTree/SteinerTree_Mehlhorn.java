package steinerTree;

import index.IndexInterface;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import main.Graph;

public class SteinerTree_Mehlhorn 
{
    public static Graph buildSteinerTree(IndexInterface index, int minimumCoreIndex, ArrayList<Integer> queryNodes, HashSet<Integer> subcore) 
    {
        if (queryNodes.size()==1)
        {
            Graph mstGraph = new Graph(subcore.size());
            mstGraph.addNode(queryNodes.get(0));
            return mstGraph;
        }
        
        Map<Integer,HashSet<Integer>> neighborCache = new HashMap<Integer,HashSet<Integer>>();

        BFPaths bfsFromDummySource = new BFPaths(index, minimumCoreIndex, queryNodes, subcore, neighborCache);
        Map<Integer,Integer> node2closestQueryNode = bfsFromDummySource.getVoronoiInfo_Mehlhorn();
        //build edges of auxiliary graph
        Map<AuxiliaryEdge,MehlhornTriple> auxiliaryEdges = new HashMap<AuxiliaryEdge,MehlhornTriple>();
        for(int u:subcore)
        {
            if(!neighborCache.containsKey(u))
            {
                neighborCache.put(u, index.getNeighbors(u, minimumCoreIndex, subcore));
            }
            for(int v:neighborCache.get(u))
            {
                if (u<v)
                {
                    int su = node2closestQueryNode.get(u);
                    int sv = node2closestQueryNode.get(v);
                    if(su != sv)
                    {
                        int du = (bfsFromDummySource.hasPathTo(u))?bfsFromDummySource.distTo(u):0;
                        int dv = (bfsFromDummySource.hasPathTo(v))?bfsFromDummySource.distTo(v):0;
                        int d = du + 1 + dv;
                        MehlhornTriple mt = new MehlhornTriple(u, v, su, sv, d);
                        AuxiliaryEdge edge = new AuxiliaryEdge(su, sv);
                        if(!auxiliaryEdges.containsKey(edge))
                        {
                            auxiliaryEdges.put(edge, mt);
                        }
                        else
                        {
                            MehlhornTriple mt_old = auxiliaryEdges.get(edge);
                            if(mt.get_d() < mt_old.get_d())
                            {
                                auxiliaryEdges.put(edge, mt);
                            }
                        }
                    }
                }
            }
        }

        // Create the weighted graph
        WeightedGraph weightedGraph = new WeightedGraph();
        // Add the query nodes
        for (int node : queryNodes)
        {
                weightedGraph.addNode(node);
        }
        // Add edges between query nodes
        for(AuxiliaryEdge edge:auxiliaryEdges.keySet())
        {
            int a = edge.getU();
            int b = edge.getV();
            int d = auxiliaryEdges.get(edge).get_d();
            weightedGraph.addEdge(a, b, d);
            weightedGraph.addEdge(b, a, d);
        } 

        // Computes the minimum spanning tree
        PrimMST mst = new PrimMST(weightedGraph);

        // Builds the graph on which apply heuristics
        //Graph mstGraph = new Graph(index.getNumberOfNodes(minimumCoreIndex));
        Graph mstGraph = new Graph(subcore.size());
        // Add nodes in each path corresponding to the edges of the MST
        Set<Integer> nodes = new HashSet<Integer>();
        for (Edge edge : mst.edges()) 
        {
            int su = edge.getFrom();
            int sv = edge.getTo();
            AuxiliaryEdge auxEdge = new AuxiliaryEdge(su, sv);
            MehlhornTriple mt = auxiliaryEdges.get(auxEdge);
            int u = mt.get_u();
            int v = mt.get_v();
            int real_su = node2closestQueryNode.get(u);
            int real_sv = node2closestQueryNode.get(v);

            nodes.add(real_su);
            nodes.add(real_sv);
            if(bfsFromDummySource.hasPathTo(u))
            {
                //add nodes on the path from dummy source to u (excluding dummy source)
                Iterable<Integer> path = bfsFromDummySource.pathTo(u);
                for(int x:path)
                {
                    if(x != -1)
                    {
                        nodes.add(x);
                    }
                }
            }
            if(bfsFromDummySource.hasPathTo(v))
            {
                //add nodes on the path from dummy source to v (excluding dummy source)
                Iterable<Integer> path = bfsFromDummySource.pathTo(v);
                for(int x:path)
                {
                    if(x != -1)
                    {
                        nodes.add(x);
                    }
                }
            }
        }

        //create induced subgraph
        for(int u:nodes)
        {
            if (!mstGraph.doesNodeExist(u)) 
            {
                mstGraph.addNode(u);
                if(!neighborCache.containsKey(u))
                {
                    neighborCache.put(u, index.getNeighbors(u, minimumCoreIndex,subcore));
                }
                for (int neighbor : neighborCache.get(u))
                {
                    if (mstGraph.doesNodeExist(neighbor)) 
                    {
                        mstGraph.addEdge(u, neighbor);
                        mstGraph.addEdge(neighbor, u);
                    }
                }
            }
        }
        
        return mstGraph;
    }
}
