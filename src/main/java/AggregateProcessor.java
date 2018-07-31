import net.imglib2.RandomAccessibleInterval;

public class AggregateProcessor implements HaralickImageProcessor {
    private BasicPreprocessor bpRows, bpCols, bpMainDiag, bpAuxDiag;

    public AggregateProcessor( int [][]window ) {
        bpRows= new BasicPreprocessor(window, RowwiseTraverser.class);
        bpCols= new BasicPreprocessor(window, ColumnwiseTraverser.class);
        bpMainDiag= new BasicPreprocessor(window, MaindiagonalTraverser.class);
        bpAuxDiag= new BasicPreprocessor(window, AuxiliarydiagonalTraverser.class);
    }

    public AggregateProcessor( final RandomAccessibleInterval<Integer> img ) {
        bpRows= new BasicPreprocessor(img, RowwiseTraverser.class);
        bpCols= new BasicPreprocessor(img, ColumnwiseTraverser.class);
        bpMainDiag= new BasicPreprocessor(img, MaindiagonalTraverser.class);
        bpAuxDiag= new BasicPreprocessor(img, AuxiliarydiagonalTraverser.class);
    }

    @Override
    public double getValue( TextureFeatures feature ) {
        return (bpRows.getValue(feature)+bpCols.getValue(feature)+bpMainDiag.getValue(feature)+bpAuxDiag.getValue(feature))/4.00;
    }
}
