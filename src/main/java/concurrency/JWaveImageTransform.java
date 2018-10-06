package concurrency;

import model.*;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedByteType;

import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.*;

public class JWaveImageTransform extends MyTransformer {
    private RandomAccessibleInterval<UnsignedByteType> src;
    private RandomAccess<UnsignedByteType> r;
    private long m,n,gridSize,K;
    private Executor executorService= Executors.newCachedThreadPool();
    private CompletionService<TaggedVector> ecs= new ExecutorCompletionService<>(executorService);
    private Class<? extends JWaveTransformOfBlock> whichClass;

    public <T extends JWaveTransformOfBlock>
    JWaveImageTransform(final RandomAccessibleInterval<UnsignedByteType> img, final long gridSize, Class<T> tClass) {
        this.src= img;
        m= src.dimension(0);
        n= src.dimension(1);
        System.out.printf("src.m= %d, src.n= %d\n",m,n);
        this.gridSize= gridSize;
        if ( gridSize > m || gridSize > n || gridSize <= 0 || 0 != ((gridSize & (gridSize-1))) )
            throw new IllegalArgumentException("gridSize must be a positive power of two");
        for ( K= 0; (1 << K) < gridSize; ++K ) ;
        r= src.randomAccess();
        whichClass= tClass;
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
                list.add(JWaveTransformFactory.getInstance(r,li,lj,gi,gj,gridSize,whichClass));
        return list;
    }
}