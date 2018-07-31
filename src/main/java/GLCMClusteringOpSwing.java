/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

import javax.swing.SwingUtilities;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.process.ImageProcessor;
import net.imagej.Data;
import net.imagej.DatasetService;
import net.imagej.ImgPlus;
import net.imagej.display.ImageDisplay;
import net.imagej.display.OverlayService;
import net.imagej.ops.AbstractOp;
import net.imagej.ops.Op;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.ItemIO;
import org.scijava.app.StatusService;
import org.scijava.command.Command;
import org.scijava.command.CommandModule;
import org.scijava.command.CommandService;
import org.scijava.command.DynamicCommand;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;
import org.scijava.util.RealRect;

@Plugin(type = Command.class,
	menuPath = "Clustering > GLCM Clustering Swing")
public class GLCMClusteringOpSwing implements Command {

	@Parameter
    private RandomAccessibleInterval<ByteType> img;

    @Parameter
	private DatasetService datasetService;

	@Parameter
	private ImageDisplay display;

	@Parameter
	private OpService ops;

	@Parameter
	private LogService log;

	@Parameter
	private UIService ui;

	@Parameter
	private CommandService cmd;

	@Parameter
	private OverlayService overlayService;

	@Parameter
	private StatusService status;

	@Parameter
	private ThreadService thread;

	//private static GLCMClusteringDialog dialog = null;
	//private static GLCMClusteringFrame dialog = null;
	private static GLCMClusteringFrame dialog = null;

	@Parameter(type = ItemIO.OUTPUT)
	private RandomAccessibleInterval<FloatType> clusteredImage;

	/**
	 * show a dialog and give the dialog access to required IJ2 Services
	 */
	@Override
	public void run() {
		SwingUtilities.invokeLater(() -> {
			if (dialog == null) {
				//dialog = new GLCMClusteringFrame();
				dialog = new GLCMClusteringFrame();
			}
			dialog.setVisible(true);
			dialog.setOps(ops);
			dialog.setLog(log);
			dialog.setStatus(status);
			dialog.setCommand(cmd);
			dialog.setThread(thread);
			dialog.setUi(ui);
			dialog.setDatasetService(datasetService);
			dialog.setDisplay(display);
			dialog.setImg(img);
			/**
			 * enables ROI (i.e. selection)
			 */
			RealRect r= overlayService.getSelectionBounds(display);
			log.info("image: "+display);
			log.info("region: +" + r.x + " +" +  r.y
					+ ", " +  r.width + " x " + r.height);
		});
	}
}

