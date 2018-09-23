
import java.util.*;

import net.imglib2.RandomAccessibleInterval;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;

import javax.xml.soap.Text;

public class BasicPreprocessor implements HaralickImageProcessor {

    private int m, n, mi, mx, H;
    private double[][] counts, probabilities;
    private double []px,py,p_xpy,p_xmy; //p_{x+y}, p_{x-y}
    private double mux,muy,sigmax,sigmay;
    private MatrixTraverser traverser;
    private double R,HX,HY,HXY,HXY1,HXY2;
    private DescriptiveStatistics statsPx= new DescriptiveStatistics();
    private DescriptiveStatistics statsPy= new DescriptiveStatistics();
    private DescriptiveStatistics statsPXY= new DescriptiveStatistics();
    private DescriptiveStatistics statsPij= new DescriptiveStatistics();
    private int [][]g;
    private int [][]seriesLengthAndBrightness;
    private Set<Pair<Integer,Integer>> presentPairs= new HashSet<>();

    private static final int W= 64;

    public <T extends MatrixTraverser>
    BasicPreprocessor( final RandomAccessibleInterval<Integer> img, Class<T> traverserImplClass ) {
        assert img.numDimensions() == 2;
        m= (int)img.max(0);
        n= (int)img.max(1);
        g= new int[m][n];
        traverser= TraverserFactory.buildTraverser(traverserImplClass,m,n);
        setUp();
    }

    public <T extends MatrixTraverser>
    BasicPreprocessor( final int[][] window, Class<T> traverserImplClass ) {
        g= new int[window.length][window[0].length];
        for ( int i= 0; i < g.length; ++i )
            for ( int j= 0; j < g[0].length; ++j )
                g[i][j]= window[i][j];
        traverser= TraverserFactory.buildTraverser(traverserImplClass,m= window.length,n= window[0].length);
        setUp();
    }

    private void setUp() {
        mi= Integer.MAX_VALUE;
        mx= Integer.MIN_VALUE;
        for ( int i= 0; i < m; ++i )
            for ( int j= 0; j < n; ++j ) {
                mi= Math.min(mi,g[i][j]);
                mx= Math.max(mx,g[i][j]);
            }
        /**
         * Trying to make H = 2^k, and H >= mx+1
         */
        if ( 0 == ((mx+1)&(mx)) ) // if mx+1 is already power of two
            H= mx+1;
        else {
            for ( H= 0; (1<<H) < (mx+1); ++H ) ;
            H= (1<<H);
        }
        assert 0 == (H & (H-1)): String.format("mi= %d, mx= %d, H is not power of two: %d\n",mi,mx,H);

        /**
         * normalisation step
         */
        //FIXME:
        if ( mx == 0 )
            mx= 1;
        for ( int i= 0; i < m; ++i )
            for ( int j= 0; j < n; ++j ) {
                // since the original matrix can be signed ints,
                // we need to shift the entire thing by "mi"
                // in order for everything to fit into [0..W-1]
                g[i][j] = (int) ((((g[i][j]-mi+0.00) / mx) * W));
                g[i][j]= Math.max(g[i][j],0);
                g[i][j]= Math.min(g[i][j],W-1);
            }
        H= W;

        seriesLengthAndBrightness= new int[Math.max(m,n)+1][H+1];

        counts = new double[H][H];
        probabilities = new double[H][H];
        px= new double[H];
        py= new double[H];
        p_xpy= new double[2*H-1];
        p_xmy= new double[H];
        populateCounts();
        calculateEntropies();
        calcSummary();
    }

    private void populateCounts() {
        /**
         * calculate co-occurrence frequency matrix
         */
        Pair<Integer,Integer> cur= null, prev;
        for ( ;traverser.hasNext(); ) {
            prev= cur; cur= traverser.next();
            if ( traverser.areAdjacent(prev,cur) ) {
                int x= g[cur.getX()][cur.getY()];
                int y= g[prev.getX()][prev.getY()];
                assert( !(x >= H || x < 0 || y >= H || y < 0) ): String.format("(%d,%d) is out of bounds",x,y);
                ++counts[x][y];
                ++counts[y][x];
            }
        }
        /**
         * calculate co-occurrence probability matrix
         */
        R= 0;
        for ( int i= 0; i < H; ++i )
            for ( int j= 0; j < H; ++j )
                R+= counts[i][j];
        for ( int i= 0; i < H; ++i )
            for ( int j= 0; j < H; ++j )
                statsPij.addValue(probabilities[i][j]= counts[i][j]/R);

        /**
         * calculate marginals
         */
        for ( int i= 0; i < H; ++i )
            for ( int j= 0; j < H; px[i]+= probabilities[i][j++] ) ;
        for ( int j= 0; j < H; ++j )
            for ( int i= 0; i < H; py[j]+= probabilities[i++][j] ) ;

        for ( int i= 0; i < H; statsPx.addValue(px[i++]) ) ;
        for ( int j= 0; j < H; statsPy.addValue(py[j++]) ) ;

        for ( int k= 0; k < 2*H-1; ++k )
            for ( int i= 0, j; i < H; ++i )
                if ( 0 <= (j= (k-i)) && j < H )
                    p_xpy[k]+= probabilities[i][j];

        for ( int k= 0; k < 2*H-1; statsPXY.addValue(p_xpy[k++]) ) ;

        for ( int k= 0; k < H; ++k )
            for ( int i= 0, j; i < H; ++i ) {
                if ( 0 <= (j= (i-k)) && j < H )
                    p_xmy[k]+= probabilities[i][j];
                if ( 0 <= (j= (i+k)) && j < H )
                    p_xmy[k]+= probabilities[i][j];
            }

        for ( int i= 0; i < H; ++i ) {
            double s= 0;
            for ( int j= 0; j < H; ++j )
                s+= probabilities[i][j];
            mux+= i*s;
        }
        for ( int j= 0; j < H; ++j ) {
            double s= 0;
            for ( int i= 0; i < H; ++i )
                s+= probabilities[i][j];
            muy+= j*s;
        }
        for ( int i= 0; i < H; ++i ) {
            double s= 0;
            for ( int j= 0; j < H; ++j )
                s+= probabilities[i][j];
            sigmax+= Math.pow(i-mux,2)*s;
        }
        for ( int j= 0; j < H; ++j ) {
            double s= 0;
            for ( int i= 0; i < H; ++i )
                s+= probabilities[i][j];
            sigmay+= Math.pow(j-muy,2)*s;
        }
        sigmax= Math.sqrt(sigmax);
        sigmay= Math.sqrt(sigmay);

        /*
         * Calculate series lengths
         */
        /* row-wise first */
        int rw, cl, nrw, ncl, nnrw, nncl;
        for ( rw= 0; rw < m; ++rw )
            for ( cl= 0; cl < n; cl= ncl ) {
                for ( ncl= cl+1; ncl < n && g[rw][cl] == g[rw][ncl]; ++ncl ) ;
                ++seriesLengthAndBrightness[ncl-cl][g[rw][cl]];
                presentPairs.add(new Pair<>(ncl-cl,g[rw][cl]));
            }

        /* col-wise second */
        for ( cl= 0; cl < n; ++cl )
            for ( rw= 0; rw < m; rw= nrw ) {
                for ( nrw= rw+1; nrw < m && g[rw][cl] == g[nrw][cl]; ++nrw ) ;
                ++seriesLengthAndBrightness[nrw-rw][g[rw][cl]];
                presentPairs.add(new Pair<>(nrw-rw,g[rw][cl]));
            }

        /* along main diagonal */
        for ( int i= 0; i < m; ++i ) {
            for ( rw= i, cl= 0; cl < n && rw >= 0; rw= nrw, cl= ncl ) {
                for ( nrw= rw, ncl= cl; nrw >= 0 && ncl < n && g[nrw][ncl] == g[rw][cl]; --nrw, ++ncl ) ;
                ++seriesLengthAndBrightness[ncl-cl][g[rw][cl]];
                presentPairs.add(new Pair<>(ncl-cl,g[rw][cl]));
            }
        }
        for ( int j= 1; j < n; ++j ) {
            for ( rw= m-1, cl= j; rw >= 0 && cl < n; rw= nrw, cl= ncl ) {
                for ( nrw= rw, ncl= cl; nrw >= 0 && ncl < n && g[nrw][ncl] == g[rw][cl]; --nrw, ++ncl ) ;
                ++seriesLengthAndBrightness[ncl-cl][g[rw][cl]];
                presentPairs.add(new Pair<>(ncl-cl,g[rw][cl]));
            }
        }

        /* along the auxiliary diagonal */
        for ( int i= m-1; i >= 0; --i ) {
            for ( rw= i, cl= 0; rw < m && cl < n; rw= nrw, cl= ncl ) {
                for ( nrw= rw, ncl= cl; nrw < m && ncl < n && g[nrw][ncl] == g[rw][cl]; ++nrw, ++ncl ) ;
                ++seriesLengthAndBrightness[ncl-cl][g[rw][cl]];
                presentPairs.add(new Pair<>(ncl-cl,g[rw][cl]));
            }
        }
        for ( int j= 1; j < n; ++j ) {
            for ( rw= 0, cl= j; rw < m && cl < n; rw= nrw, cl= ncl ) {
                for ( nrw= rw, ncl = cl; nrw < m && ncl < n && g[nrw][ncl] == g[rw][cl]; ++nrw, ++ncl ) ;
                ++seriesLengthAndBrightness[ncl-cl][g[rw][cl]];
                presentPairs.add(new Pair<>(ncl-cl,g[rw][cl]));
            }
        }
    }

    private void calculateEntropies() {
        HX= HY= HXY= HXY1= HXY2= 0;
        for ( int i= 0; i < H; ++i )
            HX+= Math.abs(px[i])>Utils.eps?px[i]*Math.log(px[i]):0;
        HX= -HX/Math.log(2);
        for ( int j= 0; j < H; ++j )
            HY+= Math.abs(py[j])>Utils.eps?py[j]*Math.log(py[j]):0;
        HY= -HY/Math.log(2);
        for ( int i= 0; i < H; ++i )
            for ( int j= 0; j < H; ++j )
                HXY+= Math.abs(probabilities[i][j])>Utils.eps?probabilities[i][j]*Math.log(probabilities[i][j]):0;
        HXY= -HXY/Math.log(2);
        for ( int i= 0; i < H; ++i )
            for ( int j= 0; j < H; ++j )
                HXY1+= Math.abs(px[i]*py[j])>Utils.eps?probabilities[i][j]*Math.log(px[i]*py[j]):0;
        HXY1= -HXY1/Math.log(2);
        for ( int i= 0; i < H; ++i )
            for ( int j= 0; j < H; ++j )
                HXY2+= Math.abs(px[i]*py[j])>Utils.eps?px[i]*py[j]*Math.log(px[i]*py[j]):0;
        HXY2= -HXY2/Math.log(2);
    }

    private Map<TextureFeatures,Double> summary;

    private void calcSummary() {
        summary= new HashMap<>();
        double s;
        int i,j,k;
        /**
         * 1. Angular Second Moment [asm]
         */
        for ( s= 0, i= 0; i < H; ++i )
            for ( j= 0; j < H; ++j )
                s+= Math.pow(probabilities[i][j],2);
        summary.put(TextureFeatures.ANGULAR_SECOND_MOMENT,s);
        /**
         * 2. Contrast [con]
         */
        for ( s= 0, k= 0; k < H; ++k )
            s+= Math.pow(k,2)*p_xmy[k];
        /*for ( s= 0, i= 0; i < H; ++i )
            for ( j= 0; j < H; ++j )
                if ( i != j )
                    s+= probabilities[i][j]*Math.pow(i-j,2);*/
        summary.put(TextureFeatures.CONTRAST,s);
        /**
         * 3. Correlation [cor]
         */
        for ( s= 0, i= 0; i < H; ++i )
            for ( j= 0; j < H; ++j )
                //s+= (i*j*probabilities[i][j]);
                s+= (i-mux)*(j-muy)*probabilities[i][j]/sigmax/sigmay;
        //s-= statsPx.getMean()*statsPy.getMean();
        //s/= (statsPx.getStandardDeviation()*statsPy.getStandardDeviation());
        summary.put(TextureFeatures.CORRELATION,s);
        /**
         * 4. Sum of squares: Variance [var]
         * FIXME: why only "i" is participating?
         */
        for ( s= 0, i= 0; i < H; ++i )
            for ( j= 0; j < H; ++j ) {
                s+= Math.pow(i-statsPij.getMean(),2)*probabilities[i][j];
            }
        summary.put(TextureFeatures.SUM_OF_SQUARES,s);
        /**
         * 5. Inverse Difference Moment [idm]
         */
        for ( s= 0, i= 0; i < H; ++i )
            for ( j= 0; j < H; ++j )
                s+= probabilities[i][j]/(1+Math.pow(i-j,2));
        summary.put(TextureFeatures.INVERSE_DIFFERENT_MOMENT,s);
        /**
         * 6. Sum Average [sav]
         */
        for ( s= 0, k= 0; k < 2*H-1; ++k )
            s+= k*p_xpy[k];
        summary.put(TextureFeatures.SUM_AVERAGE,s);

        /**
         * 8. Sum Entropy [sen]
         */
        for ( s= 0, k= 0; k < 2*H-1; ++k ) {
            double arg= Math.max(Utils.eps,p_xpy[k]);
            s+= Math.log(arg)*arg;
        }
        summary.put(TextureFeatures.SUM_ENTROPY,s/Math.log(2));
        /**
         * 7. Sum Variance [sva]
         */
        for ( s= 0, k= 0; k < 2*H-1; ++k )
            s+= Math.pow(k-summary.get(TextureFeatures.SUM_ENTROPY),2)*p_xpy[k];
        summary.put(TextureFeatures.SUM_VARIANCE,s);

        /**
         * 9. Entropy [ent]
         */
        summary.put(TextureFeatures.ENTROPY,HXY);
        /**
         * 10. Difference Variance [dva]
         */
        summary.put(TextureFeatures.DIFFERENCE_VARIANCE,statsPXY.getVariance());
        /**
         * 11. Difference Entropy [den]
         */
        for ( s= 0, k= 0; k < H; ++k ) {
            double arg= Math.max(Utils.eps,p_xmy[k]);
            s += arg*Math.log(arg);
        }
        summary.put(TextureFeatures.DIFFERENCE_ENTROPY,-s/Math.log(2));
        /**
         * 12. Information Measures of Correlation
         */
        summary.put(TextureFeatures.F12,(summary.get(TextureFeatures.ENTROPY)-HXY1)/Math.max(HX,HY));
        /**
         * 13. Information Measures of Correlation
         */
        summary.put(TextureFeatures.F13,Math.sqrt(1-Math.exp(-2*(HXY2-summary.get(TextureFeatures.ENTROPY)))));
        /**
         * 14. Maximal Correlation Coefficient
         * TODO: use JScience library
         * FIXME: we are only putting 0.00, for now
         */
        summary.put(TextureFeatures.MAXIMAL_CORRELATION_COEFFICIENT,0.00);

        double Total= 0;
        for ( k= 0; k <= m && k <= n; ++k )
            for ( int t= 0; t <= H; ++t )
                Total+= seriesLengthAndBrightness[k][t];
        /*
         * 15. Inverse Moment
         */
        double T14= 0;
        for ( Pair<Integer,Integer> pr: presentPairs ) {
            k= pr.getX(); int t= pr.getY();
            T14+= (seriesLengthAndBrightness[k][t]+0.00)/k/k;
        }
        summary.put(TextureFeatures.INVERSE_DIFFERENT_MOMENT,T14/Total);

        /*
         * 16. Moments
         */
        double T15= 0;
        for ( Pair<Integer,Integer> pr: presentPairs ) {
            k = pr.getX();
            int t = pr.getY();
            T15 += k * k * (seriesLengthAndBrightness[k][t] + 0.00);
        }
        summary.put(TextureFeatures.MOMENT,T15/Total);

        /*
         * 17. Non-homogeneity of Brightness
         */
        double T16= 0;
        for ( Pair<Integer,Integer> pr: presentPairs ) {
            k = pr.getX();
            int t = pr.getY();
            T16 += Math.pow(seriesLengthAndBrightness[k][t], 2);
        }
        summary.put(TextureFeatures.NON_HOMOGENEITY_OF_BRIGHTNESS,T16/Total);

        /*
         * 18. Non-homogeneity of Series Lengths
         */
        double T17= 0;
        /*
        for ( int t= 0; t <= H; ++t ) {
            double ss= 0.00;
            for ( k= 0; k <= m && k <= n; ++k )
                ss+= seriesLengthAndBrightness[k][t];
            T17+= Math.pow(ss,2);
        }
        */
        summary.put(TextureFeatures.NON_HOMOGENEITY_OF_SERIES_LENGTH,T17/Total);

        /*
         * 19. Share of the Image in Runs
         */
        double T18= 0.00;
        for ( Pair<Integer,Integer> pr: presentPairs ) {
            k = pr.getX();
            int t = pr.getY();
            T18 += seriesLengthAndBrightness[k][t] * k;
        }
        summary.put(TextureFeatures.SHARE_OF_IMAGE_IN_SERIES,Total/T18);

        assert summary.size() == TextureFeatures.values().length;
    }

    @Override
    public double getValue( TextureFeatures feature ) {
        assert summary.containsKey(feature);
        return summary.get(feature);
    }

}

