package model;

import concurrency.HadamardTransform01;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.clustering.Clusterable;

public class RealVector2Clusterable implements Clusterable {
    private HadamardTransform01.TaggedVector src;
    private double[] points;
    public RealVector2Clusterable( HadamardTransform01.TaggedVector r ) {
        this.src= r;
        points= src.toArray();
    }
    public int getX() {
        return src.getX();
    }
    public int getY() {
        return src.getY();
    }
    @Override
    public double[] getPoint() {
        return points;
    }
}
