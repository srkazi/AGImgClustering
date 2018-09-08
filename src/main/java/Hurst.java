import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

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
        if ( Math.abs(R) < 1e-13 )
            return Double.NaN;
        return Math.log10(R/S);
    }
}

