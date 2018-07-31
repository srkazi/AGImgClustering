import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import java.util.List;


public class DBSCANImageClusterer extends ImageClusterer<UnsignedByteType> {
    private DBSCANClusterer<AnnotatedPixelWrapper> dbscanClusterer;
    private int mask;

    //FIXME: make eps adjustable from the UI
    //FIXME: make minPts adjustable from the UI
    public DBSCANImageClusterer( int flag, final RandomAccessibleInterval<UnsignedByteType> img, double eps, int minPts, DistanceMeasure measure ) {
        super(img);
        mask= flag;
        dbscanClusterer= new DBSCANClusterer<>(eps,minPts,measure==null?new EuclideanDistance():measure);
    }

    public List<Cluster<AnnotatedPixelWrapper>> cluster() {
        return dbscanClusterer.cluster( Utils.annotateWithSlidingWindow(mask,g,Utils.DEFAULT_WINDOW_SIZE) );
    }
}

