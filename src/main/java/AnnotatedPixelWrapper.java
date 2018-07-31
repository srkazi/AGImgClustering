
import org.apache.commons.math3.ml.clustering.Clusterable;

public class AnnotatedPixelWrapper implements Clusterable {
    private Pair<Integer,Integer> location;
    private double []features;

    public Pair<Integer, Integer> getLocation() {
        return location;
    }

    public AnnotatedPixelWrapper(Pair<Integer,Integer> cellLocation, double []features ) {
        this.location= cellLocation;
        this.features= features;
    }

    /**
     * Gets the n-dimensional point.
     *
     * @return the point array
     */
    @Override
    public double[] getPoint() {
        return features;
    }
}

