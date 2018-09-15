import org.apache.commons.math3.fitting.leastsquares.*;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.optim.ConvergenceChecker;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.apache.commons.math3.util.Incrementor;
import org.apache.commons.math3.util.Pair;

import java.util.Arrays;

public class Hurst {
    public static double apply( int [][]window ) {
        int m= window.length, n= window[0].length, i,j,k= 0;
        double []x= new double[m*n];
        DescriptiveStatistics stat= new DescriptiveStatistics();
        for ( i= 0; i < m; ++i )
            for ( j= 0; j < n; ++j )
                stat.addValue(x[k++]= window[i][j]);
        double mn= stat.getMean(), miz= Double.MAX_VALUE, maz= Double.MIN_VALUE;
        for ( k= 0, i= 0; i < m; ++i )
            for ( j= 0; j < n; ++j, ++k ) {
                x[k]-= mn;
                if ( k >= 1 )
                    x[k]+= x[k-1];
                miz= Math.min(miz,x[k]);
                maz= Math.max(maz,x[k]);
            }
        double R= maz-miz, S= stat.getStandardDeviation();
        //if ( Math.abs(R) < 1e-13 )
          //  return Double.NaN;
        return Math.log10(R/S);
    }
    public static double apply( int []nums ) {
        return apply(nums,0,nums.length-1);
    }
    public static double apply( int []nums, int left, int right ) {
        int m= nums.length, i,k= 0;
        double []x= new double[right-left+1];
        DescriptiveStatistics stat= new DescriptiveStatistics();
        for ( i= left; i <= right; ++i )
            stat.addValue(x[k++]= nums[i]);
        double mn= stat.getMean(), miz= Double.MAX_VALUE, maz= Double.MIN_VALUE;
        for ( k= 0; k < m; ++k ) {
                x[k]-= mn;
                if ( k >= 1 )
                    x[k]+= x[k-1];
                miz= Math.min(miz,x[k]);
                maz= Math.max(maz,x[k]);
            }
        double R= maz-miz, S= stat.getStandardDeviation();
        //if ( Math.abs(R) < 1e-13 )
          //  return Double.NaN;
        return R/S;
    }

    private static double calc_by_chunks( final int []time_series, final int left, final int right, final int block_size ) {
        DescriptiveStatistics stat= new DescriptiveStatistics();
        for ( int i= left, j; (j= i+block_size-1) <= right; i+= block_size )
            stat.addValue(apply(time_series,i,j));
        return stat.getMean();
    }

    public static double estimate( final int [][]window ) {
        int m= window.length, n= window[0].length,i,j,k;
        int []time_series= new int[m*n];
        for ( k= 0, i= 0; i < m; ++i )
            for ( j= 0; j < n; ++j )
                time_series[k++]= window[i][j];
        return estimate(time_series,0,m*n-1);
    }

    public static double estimate( final int []time_series, final int left, final int right ) {
        int i= 0,j,k,n= right-left+1, m= n/5-3+1;
        final double []observedExpectations= new double[m];
        for ( k= 3; k <= n/5; ++k ) {
            observedExpectations[i++]= calc_by_chunks(time_series,left,right,k);
        }
        assert i == observedExpectations.length;

        MultivariateJacobianFunction model= new MultivariateJacobianFunction() {
            @Override
            public Pair<RealVector, RealMatrix> value( final RealVector point ) {
                double logC= point.getEntry(0), HH= point.getEntry(1);
                RealVector value= new ArrayRealVector(observedExpectations.length);
                RealMatrix jacobian= new Array2DRowRealMatrix(observedExpectations.length,2);
                for ( int t= 0; t < observedExpectations.length; ++t ) {
                    double L= Math.log(t+3);
                    double modelI= Math.pow(observedExpectations[t]-logC-HH*L,2);
                    value.setEntry(t,modelI);
                    //link: http://commons.apache.org/proper/commons-math/userguide/leastsquares.html
                    jacobian.setEntry(t,0,-2*(observedExpectations[t]-logC-HH*L));
                    jacobian.setEntry(t,1,-2*L*(observedExpectations[t]-logC-HH*L));
                }
                return new Pair<>(value,jacobian);
            }
        };

        double []prescribedDistances= new double[observedExpectations.length];
        Arrays.fill(prescribedDistances,0.00);

        LeastSquaresProblem problem= new LeastSquaresBuilder().
                                     start( new double[] {0.00,0.50} ).
                                     model(model).
                                     target(prescribedDistances).
                                     lazyEvaluation(false).
                                     maxEvaluations(1000).
                                     maxIterations(1000).build();
        LeastSquaresOptimizer.Optimum optimum= new LevenbergMarquardtOptimizer().optimize(problem);
        double logC= optimum.getPoint().getEntry(0), HH= optimum.getPoint().getEntry(1);
        System.out.println(String.format("C= %.2f, H= %.2f",Math.exp(logC),HH));
        return HH;
    }
}

