import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.ByteType;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.FuzzyKMeansClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import java.util.List;

public class FuzzyKMeansImageClusterer extends ImageClusterer<ByteType> {
    private FuzzyKMeansClusterer<AnnotatedPixelWrapper> fuzzyKMeansClusterer;

    //FIXME: get the fuzziness value from the UI
    public FuzzyKMeansImageClusterer( final RandomAccessibleInterval<ByteType> img, int k, double fuzziness, DistanceMeasure measure ) {
        super(img);
        fuzzyKMeansClusterer= new FuzzyKMeansClusterer<>(k,fuzziness);
    }

    public List<CentroidCluster<AnnotatedPixelWrapper>> cluster() {
        return fuzzyKMeansClusterer.cluster( Utils.annotateWithSlidingWindow(g,Utils.DEFAULT_WINDOW_SIZE) );
    }
}

