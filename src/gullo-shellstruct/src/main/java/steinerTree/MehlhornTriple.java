
package steinerTree;


public class MehlhornTriple implements Comparable
{
    private int u;
    private int v;
    private int su;
    private int sv;
    private int d;
    
    public MehlhornTriple(int u, int v, int su, int sv, int d)
    {
        this.u = u;
        this.v = v;
        this.d = d;
        
        this.su = (su<sv)?su:sv;
        this.sv = (su>sv)?su:sv;
    }

    public int get_u() 
    {
        return u;
    }

    public int get_v() 
    {
        return v;
    }

    public int get_su() 
    {
        return su;
    }

    public int get_sv() 
    {
        return sv;
    }

    public int get_d() 
    {
        return d;
    }
    
    @Override
    public boolean equals(Object o)
    {
        MehlhornTriple mt = (MehlhornTriple)o;
        if(su == mt.su && sv==mt.sv && d == mt.d)
        {
            return true;
        }
        
        return false;
    }
    
    @Override
    public int compareTo(Object o)
    {
        MehlhornTriple mt = (MehlhornTriple)o;
        
        if(d < mt.d)
        {
            return -1;
        }
        if(d > mt.d)
        {
            return 1;
        }
        
        if(su < mt.su)
        {
            return -1;
        }
        if(su > mt.su)
        {
            return 1;
        }
        
        if(sv < mt.sv)
        {
            return -1;
        }
        if(sv > mt.sv)
        {
            return 1;
        }
        
        return 0;
    }
}
