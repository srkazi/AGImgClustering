import java.util.ArrayList;
import java.util.List;

public class Utils {
    public static final double eps= 1e-7;
    public static final String RESOURCES_DIRECTORY = "/home/sj/IdeaProjects/TextureAnalysisClustering/Resources/";
    public static final int DEFAULT_WINDOW_SIZE= 9;
    public static final int NUM_TRIALS= 9;
    public static final double DEFAULT_FUZZINESS = 2.00;
    public static final int DEFAULT_NUMBER_OF_CLUSTERS = 3;
    public static final int DEFAULT_ITERS = 13;
    public static final int DEFAULT_MIN_TS = 3;
    public static final int DEFAULT_SIZE = 0x80;

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
    public static List<AnnotatedPixelWrapper> annotateWithSlidingWindow( int [][]g, int slidingWindowSize ) {
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
        for ( int i= 0; i < DEFAULT_SIZE; ++i )
            for ( int j= 0; j < DEFAULT_SIZE; ++j ) {
                System.out.println(String.format("[%d,%d] sliding window for (%d,%d)",m,n,i,j));
                for ( int x= 0, ni= i-sz; ni <= i+sz; ++ni, ++x )
                    for ( int y= 0, nj= j-sz; nj <= j+sz; ++nj, ++y ) {
                        assert x < window.length && y < window[0].length;
                        //window[x][y] = 0 <= ni && ni < m && 0 <= nj && nj < n ? g[ni][nj] : 0;
                        window[x][y] = 0 <= ni && ni < DEFAULT_SIZE && 0 <= nj && nj < DEFAULT_SIZE ? g[ni][nj] : 0;
                    }
                AnnotatedPixelWrapper wrapper= new AnnotatedPixelWrapper(new Pair<>(i,j),calcFeatures(window));
                res.add(wrapper);
            }
        return res;
    }

    private static double[] calcFeatures( int[][] window ) {
        double []features= new double[TextureFeatures.values().length];
        HaralickImageProcessor processor= new AggregateProcessor(window);
        //TODO: rewrite using Stream syntax
        int k= 0;
        for ( TextureFeatures x: TextureFeatures.values() )
            features[k++]= processor.getValue(x);
        return features;
    }
}
