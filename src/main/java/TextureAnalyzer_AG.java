/* [Haralick Textural Features](http://earlglynn.github.io/RNotes/package/EBImage/Haralick-Textural-Features.html)
 */
import ij.*;
import ij.plugin.filter.PlugInFilter;
import ij.process.*;

import java.util.*;
import ij.gui.*;
import ij.measure.ResultsTable;

public class TextureAnalyzer_AG implements PlugInFilter {

    private static final int STEP= 1, oo= (1<<30);
    private static int step;
    private ResultsTable resultsTable;

    private Map<TextureFeatures,Boolean> checked= null;

    public int setup( String arg, ImagePlus imp ) {
        if ( arg.equals("about") ) {
            showAbout();
            return DONE;
        }
        if ( imp != null && !showDialog() )
            return DONE;
        return DOES_8G+DOES_STACKS+SUPPORTS_MASKING;
        //return DOES_ALL+ROI_REQUIRED;
    }

    void showAbout() {
        IJ.showMessage("About TextureAnalyzerAG:...",
                "This plugin calculates various texture parameters.");
    }

    public void run( ImageProcessor ip ) {
        //TODO
        int m= ip.getHeight(), n= ip.getWidth();
    }

    boolean showDialog() {
        GenericDialog gd= new GenericDialog("Textural Features");
        gd.addMessage("Textural features\n"+" based in Gray-Level-Correlation-Matrices");
        gd.addNumericField("Enter the size of the step in pixels", step, 0);

        gd.addMessage("Select the features:");
        if ( checked == null )
            checked= new HashMap<TextureFeatures,Boolean>();
        for ( TextureFeatures f: TextureFeatures.values() ) {
            if ( !checked.containsKey(f) )
                checked.put(f,Boolean.valueOf(false));
            gd.addCheckbox(f.getDescription(),checked.get(f));
        }

        gd.addCheckbox("row",true);
        gd.addCheckbox("col",false);
        gd.addCheckbox("diag1",false);
        gd.addCheckbox("diag2",false);

        gd.showDialog();
        if ( gd.wasCanceled() )
            return false ;
        step= (int)gd.getNextNumber();
        for ( TextureFeatures f: TextureFeatures.values() )
            checked.put(f,Boolean.valueOf(gd.getNextBoolean()));

        return true ;
    }
}
