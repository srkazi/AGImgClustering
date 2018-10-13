import model.RealVector2Clusterable;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class Utils {
    public static final double eps= 1e-7;
    public static final String RESOURCES_DIRECTORY = "/home/sj/IdeaProjects/TextureAnalysisClustering/Resources/";
    public static final int DEFAULT_WINDOW_SIZE= 9;
    public static final int NUM_TRIALS= 9;
    public static final double DEFAULT_FUZZINESS = 2.00;
    public static final int DEFAULT_NUMBER_OF_CLUSTERS = 3;
    public static final int DEFAULT_ITERS = (1<<17);
    public static final int DEFAULT_MIN_TS = 3;
    public static final int DEFAULT_SIZE = 0x200;
    public static final int DEFAULT_RESCALE_FACTOR = 1;
    public static final int DEFAULT_GRID_SIZE = 32;

    /*
    * This is useless code, already served by the implementation-classes themselves
    public static <T extends MatrixTraverser> boolean areAdjacent( Pair<Integer,Integer> a, Pair<Integer,Integer> b, Class<T> cl ) {
        if ( cl == RowwiseTraverser.class ) {
            boolean res= a.getX() == b.getX();
        }
        if ( cl == ColumnwiseTraverser.class ) {
            boolean res= a.getY() == b.getY();
        }
        if ( cl == MaindiagonalTraverser.class ) {
            boolean res= a.getX()+a.getY() == b.getX()+b.getY();
        }
        if ( cl == AuxiliarydiagonalTraverser.class ) {
            boolean res= a.getX()-a.getY() == b.getX()-b.getY();
        }
        return false ;
    }
    */
    public static List<AnnotatedPixelWrapper> annotateWithSlidingWindow( int flag, int [][]g, int slidingWindowSize ) {
        int m,n;
        List<AnnotatedPixelWrapper> res= new ArrayList<>();
        try {
            m = g.length;
            n = g[0].length;
        } catch ( Exception e ) {
            throw new RuntimeException("the matrix of the image is empty"+e.getMessage(),e.getCause());
        }
        assert (slidingWindowSize & 1) == 1: String.format("Sliding window size must be odd, %d supplied",slidingWindowSize);
        int sz= slidingWindowSize/2;
        int [][]window= new int[slidingWindowSize][slidingWindowSize];
        for ( int i= 0; i < Math.min(m,DEFAULT_SIZE); ++i )
            for ( int j= 0; j < Math.min(n,DEFAULT_SIZE); ++j ) {
                //System.out.println(String.format("[%d,%d] sliding window for (%d,%d)",m,n,i,j));
                for ( int x= 0, ni= i-sz; ni <= i+sz; ++ni, ++x )
                    for ( int y= 0, nj= j-sz; nj <= j+sz; ++nj, ++y ) {
                        //System.out.printf("x= %d, y= %d, m= %d, n= %d\n",x,y,m,n);
                        assert x < window.length && y < window[0].length: String.format("[%d %d] is not in [%d %d]",x,y,m,n);
                        //window[x][y] = 0 <= ni && ni < m && 0 <= nj && nj < n ? g[ni][nj] : 0;
                        try {
                            window[x][y] = 0 <= ni && ni < Math.min(m, DEFAULT_SIZE) && 0 <= nj && nj < Math.min(n, DEFAULT_SIZE) ? g[ni][nj] : 0;
                        } catch ( Exception e ) {
                            System.out.println(e.getMessage());
                            e.printStackTrace();
                            throw new RuntimeException(e);
                        }
                    }
                //System.out.println("[entering] calcFeatures()");
                AnnotatedPixelWrapper wrapper= new AnnotatedPixelWrapper(new Pair<>(i,j),calcFeatures(flag,window));
                //System.out.println("[done] calcFeatures()");
                res.add(wrapper);
            }
        System.out.println("Annotation complete");
        return res;
    }

    private static double[] calcFeatures( int flag, int[][] window ) {
        double []features= new double[Integer.bitCount(flag)];
        HaralickImageProcessor processor= new AggregateProcessor(window);
        //TODO: rewrite using Stream syntax
        int k= 0, j= 0;
        for ( TextureFeatures x: TextureFeatures.values() ) {
            if ((flag & (1 << k)) != 0)
                features[j++] = processor.getValue(x);
            ++k;
        }
        return features;
    }

    public static int MASK( int logh ) {
        return (1<<logh)-1;
    }

    public static
    List<RealVector2Clusterable> normalize( List<RealVector2Clusterable> pts ) {
        for ( RealVector2Clusterable entry: pts ) {
            double[] p= entry.getPoint();
            double sum= 0;
            for ( int i= 0; i < p.length; ++i )
                sum+= p[i];
            if ( Math.abs(sum) < 1e-9 ) {
                throw new IllegalStateException("[dbscan normalization] all the components of a vector are zero");
            }
            for ( int i= 0; i < p.length; ++i )
                entry.setComponent(i,p[i]/sum);
        }
        return pts;
    }
}
