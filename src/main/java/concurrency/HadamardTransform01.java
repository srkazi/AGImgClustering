package concurrency;

import kz.ag.HadamardMatrix;
import model.RealVector2Clusterable;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.Clusterable;

import java.util.*;
import java.util.concurrent.*;

public class HadamardTransform01 {

    private RandomAccessibleInterval<UnsignedByteType> src;
    private long m,n,gridSize,K;
    private Executor executorService= Executors.newCachedThreadPool();
    private CompletionService<TaggedVector> ecs= new ExecutorCompletionService<>(executorService);
    private RealMatrix hadamardMatrix, thm;

    public class TaggedVector extends ArrayRealVector {
        private int x,y;

        public TaggedVector(int size) {
            super(size);
        }

        public int getX() {
            return x;
        }
        public void setX(int x) {
            this.x = x;
        }
        public int getY() {
            return y;
        }
        public void setY(int y) {
            this.y = y;
        }
    }

    public HadamardTransform01( final RandomAccessibleInterval<UnsignedByteType> img, final long gridSize ) {
        this.src= img;
        m= src.dimension(0);
        n= src.dimension(1);
        System.out.printf("src.m= %d, src.n= %d\n",m,n);
        this.gridSize= gridSize;
        if ( gridSize > m || gridSize > n || gridSize <= 0 || 0 != ((gridSize & (gridSize-1))) )
            throw new IllegalArgumentException("gridSize must be a positive power of two");
        for ( K= 0; (1 << K) < gridSize; ++K ) ;
        this.hadamardMatrix= HadamardMatrix.getInstance(false,(int)gridSize,(int)gridSize);
        this.thm= HadamardMatrix.getInstance(true,(int)gridSize,(int)gridSize);
    }

    public List<RealVector2Clusterable> transform()
        throws InterruptedException, ExecutionException {
        List<RealVector2Clusterable> dataPoints= new LinkedList<>();
        Collection<Callable<TaggedVector>> list= prepareCallables((int)gridSize);
        for ( Callable<TaggedVector> s: list )
            ecs.submit(s);
        for ( int i= 0; i < list.size(); ++i ) {
            TaggedVector r= ecs.take().get();
            if ( r != null )
                dataPoints.add(new RealVector2Clusterable(r));
        }
        return dataPoints;
    }

    private Collection<Callable<TaggedVector>> prepareCallables( final int gridSize ) {
        List<Callable<TaggedVector>> list= new LinkedList<>();
        int bm= (int)(m/gridSize), bn= (int)(n/gridSize);
        for ( int bi= 0, li, gi; (gi= (li= bi*gridSize)+gridSize-1) < m; ++bi )
            for ( int bj= 0, lj, gj; (gj= (lj= bj*gridSize)+gridSize-1) < n; ++bj )
                list.add(new MyWaveletTransformOfBlock(li,lj,gi,gj));
        return list;
    }

    private class MyWaveletTransformOfBlock implements Callable<TaggedVector> {

        private RandomAccess<UnsignedByteType> r= src.randomAccess();
        private RealMatrix rawMatrix;
        private int x0,y0,x1,y1;

        MyWaveletTransformOfBlock(int x0, int y0, int x1, int y1 ) {
            rawMatrix= new Array2DRowRealMatrix(x1-x0+1,y1-y0+1);
            this.x0= x0; this.x1= x1; this.y0= y0; this.y1= y1;
        }

        @Override
        public TaggedVector call() throws Exception {
            long []p= new long[3];
            for ( int i= 0, x= x0; x <= x1; ++x, ++i )
                for ( int j= 0, y= y0; y <= y1; ++y, ++j ) {
                    p[0]= x; p[1]= y; p[2]= 0;
                    r.setPosition(p);
                    if ( r.get().get() != 0 )
                        rawMatrix.setEntry(i,j,r.get().get());
                }
            TaggedVector result= new TaggedVector((x1-x0+1)*(y1-y0+1));
            result.setX((int)(x0/gridSize));
            result.setY((int)(y0/gridSize));
            RealMatrix rm= hadamardMatrix.multiply(rawMatrix).multiply(thm);
            for ( int k= 0, i= 0; i < x1-x0+1; ++i )
                for ( int j= 0; j < y1-y0+1; ++j )
                    if ( Math.abs(rm.getEntry(i,j)) > 1e-7 )
                        result.setEntry(k++,rm.getEntry(i,j));
            return result;
        }
    }
}

