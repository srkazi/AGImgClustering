import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedByteType;

import java.util.ArrayList;
import java.util.List;

public class Gridifier {
    //TODO: concurrency is overkill here?
    public static List<Pair<AxisAlignedRectangle,Double>> gridify( final int gridSize,
                                                             final RandomAccessibleInterval<UnsignedByteType> img )  {
        long[]n= {img.dimension(0),img.dimension(1)};
        int [][][][]windows= new int[(int)(n[0]/gridSize+1)][(int)(n[1]/gridSize+1)][gridSize][gridSize];
        RandomAccess<UnsignedByteType> r= img.randomAccess();
        int []p= new int[3];
        for ( long i= 0; i < n[0]; ++i )
            for ( long j= 0; j < n[1]; ++j ) {
                long bi= i/gridSize, bj= j/gridSize;
                p[0]= (int)i; p[1]= (int)j; p[2]= 0;
                r.setPosition(p);
                windows[(int)bi][(int)bj][(int)(i%gridSize)][(int)(j%gridSize)]= r.get().get();
            }
        List<Pair<AxisAlignedRectangle,Double>>res= new ArrayList<>();
        for ( int i= 0; i < (int)(n[0]/gridSize); ++i )
            for ( int j= 0; j < (int)(n[1]/gridSize); ++j ) {
                res.add(new Pair<>(new AxisAlignedRectangle(i*gridSize,j*gridSize,(i+1)*gridSize-1,(j+1)*gridSize-1),
                        Hurst.apply(windows[i][j])));
            }
        return res;
    }
}

