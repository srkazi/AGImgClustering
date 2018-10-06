package model;

import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class JWaveTransformFactory {
    public static <T extends JWaveTransformOfBlock> JWaveTransformOfBlock
    getInstance(final RandomAccess<UnsignedByteType> r, int x0, int y0, int x1, int y1, int gridSize, Class<T> tClass ) {
        if ( tClass.equals(JWaveHaar.class) )
            return new JWaveHaar(r,x0,y0,x1,y1,gridSize);
        if ( tClass.equals(JWaveDaubechies.class) )
            return new JWaveDaubechies(r,x0,y0,x1,y1,gridSize);
        if ( tClass.equals(JWaveLegendre.class) )
            return new JWaveLegendre(r,x0,y0,x1,y1,gridSize);
        if ( tClass.equals(Discrete.class) )
            return new Discrete(r,x0,y0,x1,y1,gridSize);
        return null ;
    }
}