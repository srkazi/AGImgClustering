
import ij.process.ImageProcessor;
import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.clustering.MultiKMeansPlusPlusClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import java.util.Collection;
import java.util.List;

public class MultiKMeansPlusPlusImageClusterer extends ImageClusterer<UnsignedByteType> {
    private MultiKMeansPlusPlusClusterer<AnnotatedPixelWrapper> multiKMeansPlusPlusClusterer;
    private int mask;

    //FIXME: make NUM_TRIALS somehow flexible, e.g. getting this value from user interface
    public MultiKMeansPlusPlusImageClusterer( int flag, final RandomAccessibleInterval<UnsignedByteType> img, int k, int numIters, int trials, DistanceMeasure measure ) {
        super(img);
        mask= flag;
        multiKMeansPlusPlusClusterer= new MultiKMeansPlusPlusClusterer<>(new KMeansPlusPlusClusterer<>(k,numIters),trials);
    }

    public List<CentroidCluster<AnnotatedPixelWrapper>> cluster() {
        System.out.printf("Starting clustering for %d %d\n",g.length,g[0].length);
        return multiKMeansPlusPlusClusterer.cluster( Utils.annotateWithSlidingWindow(mask,g,Utils.DEFAULT_WINDOW_SIZE) );
    }
}

