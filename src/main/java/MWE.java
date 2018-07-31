import ij.IJ;
import ij.ImagePlus;
import ij.plugin.Duplicator;
import io.scif.img.IO;
import net.imagej.ImageJ;
import net.imglib2.RandomAccess;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.FloatType;

import java.util.Random;

public class MWE {
    public MWE() {
		Img<UnsignedByteType> img = ArrayImgs.unsignedBytes(64, 64, 3);
		RandomAccess<UnsignedByteType> ra = img.randomAccess();

		Random random = new Random();

		String []colors= {"00293C","1E656D","F1F3CE","F62A00"};
		int currentColorIdx= 0;

		long[] position = new long[3];
		for (int x = 0; x < img.dimension(0); x ++) {
			for (int y = 0; y < img.dimension(1); y ++) {
				position[0] = x;
				position[1] = y;

				int redChannel= Integer.parseInt(colors[currentColorIdx].substring(0,2),16);
				int greenChannel= Integer.parseInt(colors[currentColorIdx].substring(2,4),16);
				int blueChannel= Integer.parseInt(colors[currentColorIdx].substring(4,6),16);

				position[2] = 0;
				ra.setPosition(position);
				ra.get().set(redChannel);
				position[2] = 1;
				ra.setPosition(position);
				ra.get().set(greenChannel);
				position[2] = 2;
				ra.setPosition(position);
				ra.get().set(blueChannel);

				currentColorIdx+= random.nextInt(4); currentColorIdx %= 4;
			}
		}

		ImagePlus imp = ImageJFunctions.wrap(img, "result");
		imp = new Duplicator().run(imp);
		imp.show();
		IJ.run("Stack to RGB", "");
	}
	public static void main(final String[] args) {
		final ImageJ ij= new ImageJ();
		ij.ui().showUI();

		//ij.launch(args);
		new MWE();
	}
}

