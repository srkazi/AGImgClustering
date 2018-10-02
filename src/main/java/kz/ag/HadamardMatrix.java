package kz.ag;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.HashMap;
import java.util.Map;

public class HadamardMatrix {
    private static final int N= 7;
    private static final Map<Integer,Array2DRowRealMatrix> map= new HashMap<>();
    private static final Map<Integer,Array2DRowRealMatrix> tmap= new HashMap<>();
    static {
        int k;
        Array2DRowRealMatrix base2= new Array2DRowRealMatrix(new double[][]{{1,1},{1,-1}},true);
        map.put(1,base2);
        for ( k= 2; k < N; ++k ) {
            Array2DRowRealMatrix basek= new Array2DRowRealMatrix(1<<k,1<<k);
            Array2DRowRealMatrix base_km1= map.get(k-1);
            RealMatrix m= base_km1.getSubMatrix(0,(1<<(k-1))-1,0,(1<<(k-1))-1);
            double [][]md= m.getData();
            basek.setSubMatrix(md,0,0);
            basek.setSubMatrix(md,0,(1<<(k-1)));
            basek.setSubMatrix(md,(1<<(k-1)),0);
            md= m.scalarMultiply(-1.00).getData();
            basek.setSubMatrix(md,(1<<(k-1)),(1<<(k-1)));
            map.put(k,basek);
        }
        for ( k= 1; k < N; ++k )
            tmap.put(k,(Array2DRowRealMatrix)map.get(k).transpose());
    }
    private static int getPower( int n ) {
        assert( n > 0 && 0 == (n&(n-1)) );
        int k;
        for ( k= 0; (1 << k) < n; ++k ) ;
        return k;
    }
    public static RealMatrix getInstance( boolean isTranspose, int rowDim, int colDim ) {
        if ( rowDim != colDim || 0 != (rowDim&(rowDim-1)) )
            throw new ExceptionInInitializerError("the matrix must be square and dimension = power of two");
        if ( rowDim >= (1<<N) )
            throw new ExceptionInInitializerError("size exceeds "+(1<<(N-1)));
        if ( !isTranspose )
            return map.get(getPower(rowDim));
        else return tmap.get(getPower(rowDim));
    }
    public static RealMatrix getInstance( boolean isTranspose, int rowDim, int colDim, boolean newCopy ) {
        if ( rowDim != colDim || 0 != (rowDim&(rowDim-1)) )
            throw new ExceptionInInitializerError("the matrix must be square and dimension = power of two");
        if ( rowDim >= (1<<N) )
            throw new ExceptionInInitializerError("size exceeds "+(1<<(N-1)));
        if ( !isTranspose )
            return newCopy ? map.get(getPower(rowDim)).copy() : map.get(getPower(rowDim));
        else
            return newCopy ? tmap.get(getPower(rowDim)).copy() : tmap.get(getPower(rowDim));
    }
}
