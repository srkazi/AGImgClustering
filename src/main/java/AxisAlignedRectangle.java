public class AxisAlignedRectangle extends Pair<Pair<Integer,Integer>,Pair<Integer,Integer>> {
    public AxisAlignedRectangle( int ... x ) {
        super(new Pair<>(x[0],x[1]),new Pair<>(x[2],x[3]));
    }

    public int x0() {
        return getX().getX();
    }
    public int x1() {
        return getY().getX();
    }

    public int y0() {
        return getX().getY();
    }
    public int y1() {
        return getY().getY();
    }
}
