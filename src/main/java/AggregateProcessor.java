import net.imglib2.RandomAccessibleInterval;

public class AggregateProcessor implements HaralickImageProcessor {
    private ConcurrentPreprocessor bpRows, bpCols, bpMainDiag, bpAuxDiag;

    public AggregateProcessor( int [][]window ) {
        bpRows= new ConcurrentPreprocessor(window, RowwiseTraverser.class);
        bpCols= new ConcurrentPreprocessor(window, ColumnwiseTraverser.class);
        bpMainDiag= new ConcurrentPreprocessor(window, MaindiagonalTraverser.class);
        bpAuxDiag= new ConcurrentPreprocessor(window, AuxiliarydiagonalTraverser.class);
    }

    public AggregateProcessor( final RandomAccessibleInterval<Integer> img ) {
        bpRows= new ConcurrentPreprocessor(img, RowwiseTraverser.class);
        bpCols= new ConcurrentPreprocessor(img, ColumnwiseTraverser.class);
        bpMainDiag= new ConcurrentPreprocessor(img, MaindiagonalTraverser.class);
        bpAuxDiag= new ConcurrentPreprocessor(img, AuxiliarydiagonalTraverser.class);
    }

    @Override
    public double getValue( TextureFeatures feature ) {
        return (bpRows.getValue(feature)+bpCols.getValue(feature)+bpMainDiag.getValue(feature)+bpAuxDiag.getValue(feature))/4.00;
    }
}
