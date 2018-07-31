import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.IntegerType;

class ImageClusterer<T extends IntegerType<T>> {
    private RandomAccessibleInterval<T> img;
    protected int[][]g;

    protected ImageClusterer( final RandomAccessibleInterval<T> img ) {
        this.img= img;
        extractMatrix();
    }

    private void extractMatrix() {
        int m= (int)img.max(0);
        int n= (int)img.max(1);
        System.out.println(String.format("[extractMatrix] m= %d, n= %d",m,n));
        final RandomAccess<T> r= img.randomAccess();
        assert r != null;
        System.out.println(String.format("[extractMatrix] obtained randomaccess"));
        g= new int[m][n];
        System.out.println(String.format("[extractMatrix] allocated matrix"));
        //System.out.println(String.format("%d %d",img.min(0),img.min(1)));
        for ( int i= 0; i < m; ++i ) {
            for ( int j = 0; j < n; ++j ) {
                int []dd= {i+(int)img.min(0),j+(int)img.min(1),0};
                try {
                    r.setPosition(dd);
                    g[i][j] = r.get().getInteger();
                } catch ( Exception ie ) {
                    System.out.println("Exception: "+ie.getClass()+" "+ie.getCause()+" "+ie.getMessage());
                }
            }
        }
        System.out.println("[Done extracting matrix]");
    }
}

