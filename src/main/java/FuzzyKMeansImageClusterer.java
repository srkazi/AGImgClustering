import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.List;

public class FuzzyKMeansImageClusterer extends ImageClusterer<UnsignedByteType> {
    private FuzzyKMeansClusterer<AnnotatedPixelWrapper> fuzzyKMeansClusterer;
    private int mask;

    //FIXME: get the fuzziness value from the UI
    public FuzzyKMeansImageClusterer( int flag, final RandomAccessibleInterval<UnsignedByteType> img, int k, double fuzziness, int iterations, DistanceMeasure measure ) {
        super(img);
        mask= flag;
        fuzzyKMeansClusterer= new FuzzyKMeansClusterer<>(k,fuzziness,iterations,measure==null?new EuclideanDistance():measure);
    }

    public List<CentroidCluster<AnnotatedPixelWrapper>> cluster() {
        return fuzzyKMeansClusterer.cluster( Utils.annotateWithSlidingWindow(mask,g,Utils.DEFAULT_WINDOW_SIZE) );
    }
}

