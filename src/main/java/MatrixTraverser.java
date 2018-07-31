
import java.util.Iterator;

public interface MatrixTraverser extends Iterator<Pair<Integer,Integer>> {
    boolean areAdjacent( Pair<Integer,Integer> a, Pair<Integer,Integer> b ) ;
}
