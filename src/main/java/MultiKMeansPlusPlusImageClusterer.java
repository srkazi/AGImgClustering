
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

public class MultiKMeansPlusPlusImageClusterer extends ImageClusterer<ByteType> {
    private MultiKMeansPlusPlusClusterer<AnnotatedPixelWrapper> multiKMeansPlusPlusClusterer;

    //FIXME: make NUM_TRIALS somehow flexible, e.g. getting this value from user interface
    public MultiKMeansPlusPlusImageClusterer( final RandomAccessibleInterval<ByteType> img, int k, DistanceMeasure measure ) {
        super(img);
        multiKMeansPlusPlusClusterer= new MultiKMeansPlusPlusClusterer<>(new KMeansPlusPlusClusterer<>(k),Utils.NUM_TRIALS);
    }

    public List<CentroidCluster<AnnotatedPixelWrapper>> cluster() {
        System.out.printf("Starting clustering for %d %d\n",g.length,g[0].length);
        return multiKMeansPlusPlusClusterer.cluster( Utils.annotateWithSlidingWindow(g,Utils.DEFAULT_WINDOW_SIZE) );
    }
}

