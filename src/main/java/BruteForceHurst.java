import net.imglib2.RandomAccess;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.fitting.leastsquares.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.regression.SimpleRegression;

import java.util.*;

public class BruteForceHurst {
    private RandomAccessibleInterval<UnsignedByteType> img;
    private RandomAccess<UnsignedByteType> r;
    private final int []p= new int[3], series;
    private final int [][]window;
    private final int m,n;
    private Map<Integer,Double> observedExpectations;
    double []prescribedDistances;

    public BruteForceHurst( final RandomAccessibleInterval<UnsignedByteType> img, final int s ) {
        this.img= img;
        this.r= this.img.randomAccess();
        this.m= s;
        this.window= new int[s][s];
        this.series= new int[s*s];
        this.n= m*m;
        this.prescribedDistances= new double[n/5-5+1];
    }

    private List<Double> list= new ArrayList<>();

    public List<Pair<AxisAlignedRectangle,Double>> gridify() {

        List<Pair<AxisAlignedRectangle,Double>> res= new ArrayList<>();
        System.out.printf("Entering gridify()");

        long []n= {img.dimension(0),img.dimension(1)};
        int i,j,k,ii,jj,ci,cj;
        double sum= 0,H,avg;
        for ( i= 0; i < n[0]/m; ++i ) {
            for ( j= 0; j < n[1]/m; ++j ) {
                for ( ci= 0, ii= i*m; ii < i*m+m; ++ii, ++ci )
                    for ( cj= 0, jj= j*m; jj < j*m+m; ++jj, ++cj ) {
                        p[0]= ii; p[1]= jj; p[2]= 0;
                        r.setPosition(p);
                        window[ci][cj]= r.get().get();
                    }
                System.out.printf("inside gridify: Hurst coefficient for block %d %d\n",i,j);
                list.add(H= calculateHurstCoefficient());
                sum+= H;
            }
        }
        for ( avg= sum/list.size(), k= 0, i= 0; i < n[0]/m; ++i )
            for ( j= 0; j < n[1]/m; ++j ) {
                H= list.get(k++);
                res.add(new Pair<>(new AxisAlignedRectangle(i*m,j*m,i*m+m-1,j*m+m-1),H));
            }
        return res;
    }

    private double calculateHurstCoefficient() {
        traverseZigzag();
        int i,j,k= 0,len= n/5-5+1;
        observedExpectations= new HashMap<>();
        //http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=56CC42C1007A4D15D127E103BA001462?doi=10.1.1.137.207&rep=rep1&type=pdf
        //for ( int t= 3; t <= 10 && (1 << t) <= n; ++t )
            //observedExpectations.put(1<<t,Math.log(calcByChunks(1<<t)));
        for ( int t= 16; t <= n; ++t )
            observedExpectations.put(t,Math.log(calcByChunks(t)));

        //link: http://commons.apache.org/proper/commons-math/userguide/leastsquares.html
        /*
        MultivariateJacobianFunction model= new MultivariateJacobianFunction() {
            @Override
            public org.apache.commons.math3.util.Pair<RealVector, RealMatrix> value(final RealVector point ) {
                double logC= point.getEntry(0), HH= point.getEntry(1);
                RealVector value= new ArrayRealVector(m);
                RealMatrix jacobian= new Array2DRowRealMatrix(m,2);
                for ( Map.Entry<Integer,Double> entry: observedExpectations.entrySet() ) {
                    int t= entry.getKey();
                    double L= Math.log(t);
                    double modelI= Math.pow(entry.getValue()-logC-HH*L,2);
                    value.setEntry(t-5,modelI);
                    jacobian.setEntry(t-5,0,-2*(entry.getValue()-logC-HH*L));
                    jacobian.setEntry(t-5,1,-2*L*(entry.getValue()-logC-HH*L));
                }
                return new org.apache.commons.math3.util.Pair<RealVector, RealMatrix> (value,jacobian);
            }
        };
        Arrays.fill(prescribedDistances,0.00);
        double []initial= {-1.00,0.50};
        LeastSquaresProblem problem= new LeastSquaresBuilder().
                                     start(initial).
                                     model(model).
                                     target(prescribedDistances).
                                     lazyEvaluation(false).
                                     maxEvaluations(100).
                                     maxIterations(100).
                                     build();

        System.out.printf("Launching the optimizer");
        LeastSquaresOptimizer.Optimum optimum= new LevenbergMarquardtOptimizer().optimize(problem);
        double logC= optimum.getPoint().getEntry(0), HH= optimum.getPoint().getEntry(1);
        System.out.println(String.format("C= %.2f, H= %.2f",Math.exp(logC),HH));
        return HH;
        */
        SimpleRegression simpleRegression= new SimpleRegression(true);
        double [][]data= new double[observedExpectations.size()][];
        k= 0;
        for ( Map.Entry<Integer,Double> entry: observedExpectations.entrySet() )
            data[k++]= new double[]{Math.log(entry.getKey()),entry.getValue()};
        simpleRegression.addData(data);
        System.out.println(String.format("C= %.2f, H= %.2f",Math.exp(simpleRegression.getIntercept()),simpleRegression.getSlope()));
        return simpleRegression.getSlope();
    }

    private double calcByChunks( int chunk_size ) {
        double sum= 0;
        int k= 0;
        for ( int i= 0, j; (j= i+chunk_size-1) < n; i+= chunk_size ) {
            sum += Hurst.apply(series, i, j); ++k;
        }
        return sum/k;
    }

    private void traverseZigzag() {
        int count= 0;
        try {
            for (int s = 0; s < m; ++s) {
                if (1 == (s & 1)) {
                    for (int i = s; i >= 0; --i)
                        series[count++] = window[i][s - i];
                } else {
                    for (int i = 0; i <= s; ++i)
                        series[count++] = window[i][s - i];
                }
            }
            for (int flip = ((m & 1) ^ 1), s = m; s < 2 * m - 1; ++s, flip ^= 1) {
                if (0 == (flip & 1)) {
                    for (int i = m - 1; m > s - i; --i)
                        series[count++] = window[i][s - i];
                } else {
                    for (int i = m - 1; m > s - i; --i)
                        series[count++] = window[s - i][i];
                }
            }
        } catch ( Exception e ) {
            System.out.println(e.getClass()+e.getMessage());
            throw new RuntimeException(e);
        }
        assert count == m*m;
    }

}