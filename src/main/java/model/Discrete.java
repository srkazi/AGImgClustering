package model;

import jwave.Transform;
import jwave.transforms.FastWaveletTransform;
import jwave.transforms.wavelets.other.DiscreteMayer;
import net.imglib2.RandomAccess;
import net.imglib2.type.numeric.integer.UnsignedByteType;

public class Discrete extends JWaveTransformOfBlock {
    public Discrete(RandomAccess<UnsignedByteType> r, int x0, int y0, int x1, int y1, int gridSize) {
        super(r, x0, y0, x1, y1, gridSize);
        transform= new Transform(new FastWaveletTransform(new DiscreteMayer()));
    }
}