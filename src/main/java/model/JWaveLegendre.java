package model;

import jwave.Transform;
import jwave.transforms.FastWaveletTransform;
import jwave.transforms.wavelets.legendre.Legendre2;
import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class JWaveLegendre extends JWaveTransformOfBlock {
    public JWaveLegendre(RandomAccess<UnsignedByteType> r, int x0, int y0, int x1, int y1, int gridSize) {
       super(r, x0, y0, x1, y1, gridSize);
       if ( gridSize != 8 && gridSize != 32 && gridSize != 128 ) {
           this.gridSize= 32;
           System.out.printf("Gridsize corrected for the Legendre transform to be 128\n");
       }
       transform= new Transform(new FastWaveletTransform(new Legendre2()));
    }
}