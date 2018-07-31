
public class Pair<U,V> {

    private U x;
    private V y;

    public Pair() { x= null; y= null; }

    public  Pair( U u, V v ) {
        x= u;
        y= v;
    }

    public U getX() {
        return x;
    }

    public V getY() {
        return y;
    }

    @Override
    public boolean equals( Object obj ) {
        if ( !(obj instanceof Pair) )
            return false ;
        Pair<U,V> o= (Pair<U,V>)(obj);
        return o.getY().equals(y) && o.getX().equals(x);
    }

    @Override
    public int hashCode() {
        return x.hashCode() & y.hashCode();
    }
}
