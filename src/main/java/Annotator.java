public class Annotator implements Runnable {

    private int [][]g;
    private int sz;
    private Pair<Integer,Integer> upperLeft, lowerRight;

    // annotate each cell of the given matrix "g"'s rectangle slice [a,c]x[b,d],
    // using the neighborhood of size "sz" centered at that cell
    public Annotator( final int [][]window, int a, int b, int c, int d, int sz ) {
        this.g= window;
        upperLeft= new Pair<>(a,b);
        lowerRight= new Pair<>(c,d);
        assert a <= c;
        assert b <= d;
        this.sz= sz;
    }
    /**
     * When an object implementing interface <code>Runnable</code> is used
     * to create a thread, starting the thread causes the object's
     * <code>run</code> method to be called in that separately executing
     * thread.
     * <p>
     * The general contract of the method <code>run</code> is that it may
     * take any action whatsoever.
     *
     * @see Thread#run()
     */
    @Override
    public void run() {
    }
}

