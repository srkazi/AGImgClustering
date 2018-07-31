public class TraverserFactory {
    public static <T extends MatrixTraverser> MatrixTraverser buildTraverser( Class<T> cl, int m, int n ) {
        if ( cl.equals(RowwiseTraverser.class) )
            return new RowwiseTraverser(m,n);
        if ( cl.equals(ColumnwiseTraverser.class) )
            return new ColumnwiseTraverser(m,n);
        if ( cl.equals(MaindiagonalTraverser.class) )
            return new MaindiagonalTraverser(m,n);
        if ( cl.equals(AuxiliarydiagonalTraverser.class) )
            return new AuxiliarydiagonalTraverser(m,n);
        return null;
    }
}
