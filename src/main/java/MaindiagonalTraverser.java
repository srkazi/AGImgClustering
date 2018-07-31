
import java.util.NoSuchElementException;

public class MaindiagonalTraverser implements MatrixTraverser {
    private int m,n,x,y,k,cnt; //(x,y) is the current cell, k == (x+y)
    private boolean flag;
    private int[][]id;

    private void reset() {
        x= y= 0; k= x+y; cnt= 0; flag= false;
    }
    /**
     * @param m: number of rows
     * @param n: number of columns
     */
    public MaindiagonalTraverser( int m, int n ) {
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
            --y; ++x;
            if ( !vc(x,y) ) {
                ++k;
                if ( flag ) {
                    y= n-1;
                    x= k-y;
                }
                else {
                    x = 0;
                    y = k - x;
                    if (!vc(x, y)) {
                        flag= true ;
                        y = n - 1;
                        x = k - y;
                    }
                }
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
        return a.getX()+a.getY() == b.getX()+b.getY();
    }

    private Pair<Integer,Integer> next( int x, int y ) {
        Pair<Integer,Integer> res= null;
        if ( vc(x,y) && id[x][y] < m*n-1 ) {
            k= y+x;
            --y; ++x;
            if ( !vc(x,y) ) {
                ++k;
                if ( flag ) {
                    y= n-1;
                    x= k-y;
                }
                else {
                    x = 0;
                    y = k - x;
                    if (!vc(x, y)) {
                        flag= true ;
                        y = n - 1;
                        x = k - y;
                    }
                }
            }
            res= new Pair<>(x,y);
        }
        assert vc(x,y);
        return res;
    }

    private boolean vc( int x, int y ) {
        return  0<=x && x<m && 0<=y && y<n;
    }
}
