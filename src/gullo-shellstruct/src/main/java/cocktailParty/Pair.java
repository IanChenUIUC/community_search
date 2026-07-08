package cocktailParty;

public class Pair implements Comparable
{
    private int id;
    private double value;
    
    public Pair(int id, double value)
    {
        this.id = id;
        this.value = value;
    }

    public int getId() {
        return id;
    }

    public double getValue() {
        return value;
    }
    
    @Override
    public boolean equals(Object o)
    {
        Pair p = (Pair)o;
        
        return p.id == this.id && p.value == this.value;
    }

    @Override
    public int compareTo(Object o) 
    {
        Pair p = (Pair)o;
        
        if(this.value < p.value)
        {
            return -1;
        }
        
        if(this.value > p.value)
        {
            return 1;
        }
        
        if(this.id < p.id)
        {
            return -1;
        }
        
        if(this.id > p.id)
        {
            return 1;
        }
        
        return 0;
    }
    
    @Override
    public int hashCode()
    {
        return new Integer(id).hashCode();
    }
    
    
}
