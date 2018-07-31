
import java.util.NoSuchElementException;

public class RowwiseTraverser implements MatrixTraverser {
    private int m,n,x,y,cnt; //(x,y) is the current cell
    private int [][]id;

    private void reset() {
        x= y= 0; cnt= 0;
    }
    /**
     * @param m: number of rows
     * @param n: number of columns
     */
    public RowwiseTraverser( int m, int n ) {
        this.m= m; this.n= n;
        reset();
        assert m >= 0 && n >= 0;
        id= new int[m][n];
        int idx= 0;
        for ( int i= 0; i < m; ++i )
            for ( int j= 0; j < n; ++j ) {
                id[x][y]= idx++;
                if ( hasNext() )
                    next();
            }
        reset();
    }

    /**
     * Returns {@code true} if the iteration has more elements.
     * (In other words, returns {@code true} if {@link #next} would
     * return an element rather than throwing an exception.)
     *
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return cnt < m*n;
    }

    /**
     * Returns the next element in the iteration.
     *
     * @return the next element in the iteration
     * @throws NoSuchElementException if the iteration has no more elements
     */
    @Override
    public Pair<Integer, Integer> next() {
        Pair<Integer,Integer> res= new Pair<>(x,y);
        ++cnt;
        if ( hasNext() ) {
            if (++y == n) {
                y = 0;
                ++x;
            }
        }
        assert vc(x,y);
        return res;
    }

    @Override
    public boolean areAdjacent(Pair<Integer, Integer> a, Pair<Integer, Integer> b) {
        if ( a == null || b == null ) return false ;
        /*Pair<Integer,Integer> na= next(a.getX(),a.getY());
        Pair<Integer,Integer> nb= next(b.getX(),b.getY());
        return na.equals(b) || nb.equals(a);*/
        return a.getX() == b.getX();
    }

    private Pair<Integer,Integer> next( int x, int y ) {
        Pair<Integer,Integer> res= null;
        if ( vc(x,y) && id[x][y] < m*n-1 ) {
            if (++y == n) {
                y = 0;
                ++x;
            }
            res= new Pair<>(x,y);
        }
        assert vc(x,y);
        return res;
    }

    private boolean vc( int x, int y ) {
        return 0<=x && x<m && 0<=y && y<n;
    }
}

