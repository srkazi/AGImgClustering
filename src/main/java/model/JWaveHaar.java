package model;

import jwave.Transform;
import jwave.transforms.FastWaveletTransform;
import jwave.transforms.wavelets.haar.Haar1;
import jwave.transforms.wavelets.haar.Haar1Orthogonal;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.RandomAccess;

public class JWaveHaar extends JWaveTransformOfBlock {
    public JWaveHaar(net.imglib2.RandomAccess<UnsignedByteType> r, int x0, int y0, int x1, int y1, int gridSize) {
        super(r,x0,y0,x1,y1,gridSize);
        transform= new Transform(new FastWaveletTransform(new Haar1Orthogonal()));
    }
}