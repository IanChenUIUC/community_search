
package steinerTree;

public class AuxiliaryEdge implements Comparable
{
    private int u;
    private int v;
    
    public AuxiliaryEdge(int u, int v)
    {
        if (u < v)
        {
            this.u = u;
            this.v = v;
        }
        else
        {
            this.v = u;
            this.u = v;
        }
    }

    public int getU() 
    {
        return u;
    }

    public int getV() 
    {
        return v;
    }
    
    @Override
    public boolean equals(Object o)
    {
        AuxiliaryEdge p = (AuxiliaryEdge)o;
        
        return p.u == this.u && p.v == this.v;
    }

    @Override
    public int compareTo(Object o) 
    {
        AuxiliaryEdge p = (AuxiliaryEdge)o;
        
        if(this.u < p.u)
        {
            return -1;
        }
        
        if(this.u > p.u)
        {
            return 1;
        }
        
        if(this.v < p.v)
        {
            return -1;
        }
        
        if(this.v > p.v)
        {
            return 1;
        }
        
        return 0;
    }
    
    @Override
    public int hashCode()
    {
        return (""+u+"X"+v).hashCode();
    }
    
    
}

