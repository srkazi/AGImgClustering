package model;

import jwave.Transform;
import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.concurrent.Callable;

public abstract class JWaveTransformOfBlock implements Callable<TaggedVector> {
    protected RealMatrix rawMatrix;
    protected int x0,y0,x1,y1,gridSize;
    protected Transform transform;

    public JWaveTransformOfBlock( final RandomAccess<UnsignedByteType> r, int x0, int y0, int x1, int y1, int gridSize ) {
        rawMatrix= new Array2DRowRealMatrix(x1-x0+1,y1-y0+1);
        this.x0= x0; this.x1= x1; this.y0= y0; this.y1= y1;
        this.gridSize= gridSize;
        long []p= new long[3];
            for ( int i= 0, x= x0; x <= x1; ++x, ++i )
                for ( int j= 0, y= y0; y <= y1; ++y, ++j ) {
                    p[0]= x; p[1]= y; p[2]= 0;
                    r.setPosition(p);
                    if ( r.get().get() != 0 )
                        rawMatrix.setEntry(i,j,r.get().get());
                }
    }

    @Override
    public TaggedVector call() throws Exception {
        TaggedVector result= new TaggedVector((x1-x0+1)*(y1-y0+1));
            result.setX((int)(x0/gridSize));
            result.setY((int)(y0/gridSize));
            RealMatrix rm= new Array2DRowRealMatrix(transform.forward(rawMatrix.getData()));
            for ( int k= 0, i= 0; i < x1-x0+1; ++i )
                for ( int j= 0; j < y1-y0+1; ++j )
                    if ( Math.abs(rm.getEntry(i,j)) > 1e-7 )
                        result.setEntry(k++,rm.getEntry(i,j));
        return result;
    }
}