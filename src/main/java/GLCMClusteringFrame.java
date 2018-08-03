import ij.IJ;
import ij.ImagePlus;
import ij.plugin.Duplicator;
import net.imagej.DatasetService;
import net.imagej.display.ImageDisplay;
import net.imagej.display.OverlayService;
import net.imagej.ops.OpService;
import net.imglib2.*;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.display.imagej.ImageJFunctions;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Cluster;
import org.apache.commons.math3.ml.distance.*;
import org.scijava.app.StatusService;
import org.scijava.command.CommandService;
import org.scijava.log.LogService;
import org.scijava.thread.ThreadService;
import org.scijava.ui.UIService;

import javax.swing.*;
import javax.swing.border.EtchedBorder;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GLCMClusteringFrame extends JFrame {

	final static boolean shouldFill = true;
	final static boolean shouldWeightX = true;
	final static boolean RIGHT_TO_LEFT = false;

	private int flag= 0;
    private OpService ops;
	private LogService log;
	private StatusService status;
	private CommandService cmd;
	private ThreadService thread;
	private UIService ui;
	private ImageDisplay display;
	private OverlayService overlayService;
	private DatasetService datasetService;
	private RandomAccessibleInterval<UnsignedByteType> img;
	private RandomAccessibleInterval<UnsignedByteType> currentSelection;

	private final JPanel contentPanel= new JPanel();
	private final JTabbedPane tabbedPane= new JTabbedPane();
	private JMenuBar bar;
    private Img<UnsignedByteType> selectedRegion;
    private DistanceMeasure selectedDistance= null;

    public GLCMClusteringFrame() {
		setBounds(100,100,300,300);
		this.setTitle("Clustering");

		/*
		try {
			UIManager.setLookAndFeel("javax.swing.plaf.nimbus.NimbusLookAndFeel");
		} catch (ClassNotFoundException e) {
			e.printStackTrace();
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (UnsupportedLookAndFeelException e) {
			e.printStackTrace();
		}
		*/
		JComponent panelForFuzzyKMeans= makeFuzzyKMeansTab();
		tabbedPane.addTab("Fuzzy K-Means",panelForFuzzyKMeans);
		JComponent panelForDBSCAN= makeDBSCANTab("DBSCAN");
		tabbedPane.addTab("DBSCAN",panelForDBSCAN);
		JComponent panelForMultiKMeansPP= makeMultiKMeansTab();
		tabbedPane.addTab("Multi-K-Means++",panelForMultiKMeansPP);

		tabbedPane.setPreferredSize(new Dimension(350,200));

		add(tabbedPane);
		tabbedPane.setTabLayoutPolicy(JTabbedPane.SCROLL_TAB_LAYOUT);
		this.pack();

		makeMenuBar();
	}

	private JComponent makeMultiKMeansTab() {
	    JPanel panel= new JPanel();
		panel.setLayout(new GridBagLayout());
		GridBagConstraints c= new GridBagConstraints();
		/*
		JLabel label= new JLabel("#of clusters",SwingConstants.RIGHT);
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 0;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.weightx= 0.3;
		panel.add(label,c);
		*/

		JFormattedTextField formattedTextField= new JFormattedTextField();
		formattedTextField.setToolTipText("must be >= 1");
		formattedTextField.setInputVerifier(new InputVerifier() {
            @Override
            public boolean verify(JComponent input) {
                try {
                    int k= Integer.parseInt(((JFormattedTextField)input).getText());
                    return k >= 1;
                } catch ( Exception e ) {
                    return false ;
                }
            }
        });
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 0;
		c.gridwidth= 4;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.LINE_END;
		c.weightx= 0.7;
		formattedTextField.setBorder( BorderFactory.createTitledBorder(
				BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "#of clusters"
		));
		panel.add(formattedTextField,c);

		/*
		JLabel label2= new JLabel("maximum #of iterations",SwingConstants.RIGHT);
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 1;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.weightx= 0.3;
		panel.add(label2,c);
		*/

		JFormattedTextField formattedTextField2= new JFormattedTextField();
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 1;
		c.gridwidth= 4;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.LINE_END;
		c.weightx= 0.7;
		formattedTextField2.setBorder( BorderFactory.createTitledBorder(
			BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "max #of iterations"
		));
		panel.add(formattedTextField2,c);

		/*
		JLabel label3= new JLabel("#of trials",SwingConstants.RIGHT);
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 2;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.weightx= 0.3;
		panel.add(label3,c);
		*/

		JFormattedTextField formattedTextField3= new JFormattedTextField();
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 2;
		c.gridwidth= 4;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.LINE_END;
		c.weightx= 0.7;
		formattedTextField3.setBorder( BorderFactory.createTitledBorder(
			BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "#of trials"
		));
		panel.add(formattedTextField3,c);

		final JButton clusterIt= new JButton("Cluster");
		clusterIt.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				thread.run( ()-> {
					int k= 3, numIters, trials;
					try {
						k= Integer.parseInt(formattedTextField.getText());
                        numIters= Integer.parseInt(formattedTextField2.getText());
                        trials= Integer.parseInt(formattedTextField3.getText());
					} catch ( NumberFormatException nfe ) {
						//throw nfe;
                        k= Utils.DEFAULT_NUMBER_OF_CLUSTERS;
                        numIters= Utils.DEFAULT_ITERS;
                        trials= Utils.NUM_TRIALS;
					}
					log.info("Read k= "+k);
					selectDistanceMeasure();
					getWindowSize();
					multiKMeansPPClustering(k,numIters,trials);
				});
			}
		});
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 3;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.CENTER;
		c.weightx= 0.7;
		panel.add(clusterIt,c);

		return panel;
	}

	private int windowSize;
    private void getWindowSize() {
        for ( Map.Entry<String,JRadioButtonMenuItem> entry: str2windowsize.entrySet() ) {
            JRadioButtonMenuItem item= entry.getValue();
            if ( item.isSelected() ) {
                windowSize= Integer.parseInt(entry.getKey().substring(0,entry.getKey().indexOf("x")));
                log.info("Selected window size: "+windowSize);
                return ;
            }
        }
        windowSize= Utils.DEFAULT_WINDOW_SIZE;
        log.info("Selected window size: "+windowSize);
    }

    public void multiKMeansPPClustering( int k, int numIters, int trials ) {
	    /*
		RealRect r= overlayService.getSelectionBounds(display);
		*/
		MultiKMeansPlusPlusImageClusterer clusterer= new MultiKMeansPlusPlusImageClusterer(flag,selectedRegion,k, numIters, trials,selectedDistance,windowSize);
		log.info("[Launching Multi-KMeans Clustering]");
		try {
            List<CentroidCluster<AnnotatedPixelWrapper>> list = clusterer.cluster();
            drawResult(list);
            log.info("[DONE Multi-KMeans Clustering]");
        } catch ( Exception e ) {
		    log.info(e.getCause());
		    log.info(e.getMessage());
		    e.printStackTrace();
        }
	}

	private void dbscanClustering( int minPts, double eps ) {
		DBSCANImageClusterer clusterer= new DBSCANImageClusterer(flag,selectedRegion,eps,minPts,selectedDistance,windowSize);
		log.info(String.format("[Launching DBSCAN Clustering with eps= %f, minPts= %d]",eps,minPts));
		try {
		    List<Cluster<AnnotatedPixelWrapper>> list= clusterer.cluster();
		    log.info("[DONE DBSCAN Clustering]");
		    drawResult(list);
		} catch ( Exception e ) {
		    log.info(e.getCause());
		    log.info(e.getMessage());
		    e.printStackTrace();
        }
	}

	private void fuzzyKMeansClustering( int k, double fuzziness, int numIterations ) {
		FuzzyKMeansImageClusterer clusterer= new FuzzyKMeansImageClusterer(flag,selectedRegion,k,fuzziness,numIterations,selectedDistance,windowSize);
		log.info("[Launching FuzzyKMeans Clustering]");
		try {
            List<CentroidCluster<AnnotatedPixelWrapper>> list = clusterer.cluster();
            log.info("[DONE FuzzyKMeans Clustering]");
            drawResult(list);
        } catch ( Exception e ) {
            log.info(e.getCause());
		    log.info(e.getMessage());
		    e.printStackTrace();
        }
	}

	private void drawResult( List<? extends Cluster<AnnotatedPixelWrapper>> list ) {
    	//ImgFactory<UnsignedByteType> imgFactory= new ArrayImgFactory<>();
		//Img<UnsignedByteType> img= imgFactory.create( new int[]{Utils.DEFAULT_SIZE,Utils.DEFAULT_SIZE,3}, new UnsignedByteType() );
		Img<UnsignedByteType> img= ArrayImgs.unsignedBytes(Utils.DEFAULT_SIZE, Utils.DEFAULT_SIZE, 3);
		RandomAccess<UnsignedByteType> r= img.randomAccess();
		String []colors= {"00293C","1E656D","F1F3CE","F62A00"};
		int currentColorIdx= 0;
		long []p= new long[3];
		for ( Cluster<AnnotatedPixelWrapper> cl: list ) {
			List<AnnotatedPixelWrapper> points= cl.getPoints();
			int redChannel= Integer.parseInt(colors[currentColorIdx].substring(0,2),16);
			int greenChannel= Integer.parseInt(colors[currentColorIdx].substring(2,4),16);
			int blueChannel= Integer.parseInt(colors[currentColorIdx].substring(4,6),16);
			for ( AnnotatedPixelWrapper apw: points ) {
				Pair<Integer,Integer> location= apw.getLocation();
				int i= location.getX(), j= location.getY();
				assert 0 <= i && i < Utils.DEFAULT_SIZE;
				assert 0 <= j && j < Utils.DEFAULT_SIZE;
				p[0]= i; p[1]= j;
				p[2]= 0; r.setPosition(p);
				r.get().set(redChannel);
				p[2]= 1; r.setPosition(p);
				r.get().set(greenChannel);
				p[2]= 2; r.setPosition(p);
				r.get().set(blueChannel);
			}
			++currentColorIdx;
		}
		//ImageJFunctions.show(img);
        ImagePlus imp= ImageJFunctions.wrap(img,"result");
		imp= new Duplicator().run(imp);
		imp.show();
		IJ.run("Stack to RGB", "");
	}

	private JComponent makeDBSCANTab( String dbscan ) {
		JPanel panel= new JPanel();
		panel.setLayout(new GridBagLayout());
		GridBagConstraints c= new GridBagConstraints();

		/*
		JLabel label= new JLabel("eps");
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 0;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.weightx= 0.3;
		label.setHorizontalAlignment(SwingConstants.RIGHT);
		label.setHorizontalTextPosition(SwingConstants.RIGHT);
		panel.add(label,c);
		*/

		JFormattedTextField formattedTextField= new JFormattedTextField();
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 0;
		c.gridwidth= 4;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.SOUTH;
		c.weightx= 0.7;
		formattedTextField.setBorder( BorderFactory.createTitledBorder(
			BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "eps"
		));
		panel.add(formattedTextField,c);

		/*
		JLabel label2= new JLabel("min. #of points",SwingConstants.RIGHT);
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 1;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.weightx= 0.3;
		label2.setHorizontalAlignment(SwingConstants.RIGHT);
		label2.setHorizontalTextPosition(SwingConstants.RIGHT);
		panel.add(label2,c);
		*/

		JFormattedTextField formattedTextField2= new JFormattedTextField();
		formattedTextField2.setInputVerifier(new InputVerifier() {
			@Override
			public boolean verify( JComponent input ) {
				try {
					int k= Integer.parseInt( ((JFormattedTextField)input).getText() );
					return 1 <= k;
				} catch ( Exception e ) {
					return false ;
				}
			}
		});
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 1;
		c.gridwidth= 4;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.LINE_END;
		c.weightx= 0.7;
		formattedTextField2.setBorder( BorderFactory.createTitledBorder(
			BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "min #of points"
		));
		panel.add(formattedTextField2,c);

		final JButton clusterIt= new JButton("Cluster");
		clusterIt.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				thread.run( ()-> {
					int minPts= 3;
					double eps;
					try {
						minPts= Integer.parseInt(formattedTextField2.getText());
						eps= Double.parseDouble(formattedTextField.getText());
					} catch ( NumberFormatException nfe ) {
						//throw nfe;
						eps= 1e-3;
						minPts= Utils.DEFAULT_MIN_TS;
					}
					log.info("Read minPts= "+minPts);
					selectDistanceMeasure();
					getWindowSize();
					dbscanClustering(minPts,eps);
				});
			}
		});
		c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 2;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.CENTER;
		c.weightx= 0.7;
		panel.add(clusterIt,c);

		return panel;

	}

	private Map<String,DistanceMeasure> str2distance= new HashMap<>();
    private Map<String,JRadioButtonMenuItem> str2button= new HashMap<>();
    private Map<String,JRadioButtonMenuItem> str2windowsize= new HashMap<>();

	protected void makeMenuBar() {
		bar= new JMenuBar();

		//TODO: create resource bundle...
		JMenu measuresMenu= new JMenu("Distance Measures");
		ButtonGroup measuresGroup= new ButtonGroup();
		//TODO: add a tooltip to each of these
		String []distances= {"Euclidean","Chebyshev","Canberra","Manhattan","Earth Mover's"};
        DistanceMeasure []dm= {new EuclideanDistance(),new ChebyshevDistance(), new CanberraDistance(), new ManhattanDistance(), new EarthMoversDistance()};
        for ( int i= 0; i < distances.length; ++i )
            str2distance.put(distances[i],dm[i]);
		for ( String x: distances ) {
			JRadioButtonMenuItem item = new JRadioButtonMenuItem(x);
			measuresGroup.add(item);
			measuresMenu.add(item);
			str2button.put(x,item);
		}
		bar.add(measuresMenu);

		JMenu windowSizeMenu= new JMenu("Window Size");
		String []windowSizes= {"3x3","5x5","7x7","9x9","11x11"};
		ButtonGroup windowSizesGroup= new ButtonGroup();
		for ( String x: windowSizes ) {
			JRadioButtonMenuItem item = new JRadioButtonMenuItem(x);
			windowSizesGroup.add(item);
			windowSizeMenu.add(item);
			str2windowsize.put(x,item);
		}
		bar.add(windowSizeMenu);

		//FIXME: only one item is selected, probably need to read this:
		// https://java-swing-tips.blogspot.com/2016/07/select-multiple-jcheckbox-in-jcombobox.html
        // https://stackoverflow.com/questions/19766/how-do-i-make-a-list-with-checkboxes-in-java-swing
		/*JMenu adjacencyTypeMenu= new JMenu("Adjacency");
		String []adjacencies= {"Rows","Columns","Main diagonal","Aux diagonal"};
		ButtonGroup adjacenciesGroup= new ButtonGroup();
		for ( String x: adjacencies ) {
			JCheckBox item = new JCheckBox(x);
			adjacenciesGroup.add(item);
			adjacencyTypeMenu.add(item);
		}
		bar.add(adjacencyTypeMenu);
		*/
		JMenu featuresMenu= new JMenu("Features");
		final JList<TextureFeatures> listOfFeatures= new JList<>(TextureFeatures.values());
		listOfFeatures.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);
		listOfFeatures.addListSelectionListener(new ListSelectionListener() {
            @Override
            public void valueChanged( ListSelectionEvent e ) {
                for (int row = 0; row < TextureFeatures.values().length; ++row ) {
                    if ( listOfFeatures.isSelectedIndex(row) ) {
                        flag|= (1<<row);
                    }
                    else flag&= ~(1<<row);
                }
                log.info("The features chosen: ");
                for ( int i= 0; i < TextureFeatures.values().length; ++i )
                    if ( 0 != (flag & (1<<i)) )
                        log.info("Chosen: "+TextureFeatures.values()[i]);
            }
        });
		featuresMenu.add(listOfFeatures);
		bar.add(featuresMenu);

		this.setJMenuBar(bar);
	}

	protected JComponent makeFuzzyKMeansTab() {
		JPanel panel= new JPanel(false);
		panel.setLayout(new GridBagLayout());
		GridBagConstraints constraints;
		/*
		JLabel label1= new JLabel("#of clusters",SwingConstants.RIGHT);
		constraints= new GridBagConstraints();
		constraints.gridx= 0;
		constraints.gridy= 0;
		constraints.gridwidth= 1;
		constraints.gridheight= 1;
		constraints.weightx= 0.3;
		panel.add(label1,constraints);
		*/

		JFormattedTextField numberOfClusters= new JFormattedTextField();
		constraints= new GridBagConstraints();
		constraints.gridx= 0;
		constraints.gridy= 0;
		constraints.gridwidth= 4;
		constraints.gridheight= 1;
		constraints.fill= GridBagConstraints.HORIZONTAL;
		constraints.anchor= GridBagConstraints.LINE_END;
		constraints.weightx= 0.7;
		numberOfClusters.setBorder( BorderFactory.createTitledBorder(
				BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "#of clusters"
		));
		numberOfClusters.setToolTipText("Must be >= 1");
		numberOfClusters.setInputVerifier(new InputVerifier() {
			@Override
			public boolean verify( JComponent input ) {
				try {
					int k= Integer.parseInt( ((JFormattedTextField)input).getText() );
					return 1 <= k;
				} catch ( Exception e ) {
					return false ;
				}
			}
		});
		panel.add(numberOfClusters,constraints);

		/*
		JLabel label2= new JLabel("fuzziness",SwingConstants.RIGHT);
		constraints= new GridBagConstraints();
		constraints.gridx= 0;
		constraints.gridy= 1;
		constraints.gridwidth= 1;
		constraints.gridheight= 1;
		constraints.weightx= 0.3;
		panel.add(label2,constraints);
		*/

		JFormattedTextField textField= new JFormattedTextField();
		textField.setInputVerifier(new InputVerifier() {
			@Override
			public boolean verify(JComponent input) {
			    try {
			    	double eps= Double.parseDouble( ((JFormattedTextField)input).getText() );
			    	return  eps > 1.00;
				} catch ( Exception e ) {
			    	return false ;
				}
			}
		});
		textField.setToolTipText("Must be > 1.00");
		constraints= new GridBagConstraints();
		constraints.gridx= 0;
		constraints.gridy= 1;
		constraints.gridwidth= 4;
		constraints.gridheight= 1;
		constraints.fill= GridBagConstraints.HORIZONTAL;
		constraints.anchor= GridBagConstraints.LINE_END;
		constraints.weightx= 0.7;
		textField.setBorder( BorderFactory.createTitledBorder(
				BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "fuzziness"
		));
		panel.add(textField,constraints);

		/*
		JLabel label3= new JLabel("#of iterations",SwingConstants.RIGHT);
		constraints= new GridBagConstraints();
		constraints.gridx= 0;
		constraints.gridy= 2;
		constraints.gridwidth= 1;
		constraints.gridheight= 1;
		constraints.weightx= 0.3;
		panel.add(label3,constraints);
		*/

		JFormattedTextField textField2= new JFormattedTextField();
		constraints= new GridBagConstraints();
		constraints.gridx= 0;
		constraints.gridy= 2;
		constraints.gridwidth= 4;
		constraints.gridheight= 1;
		constraints.fill= GridBagConstraints.HORIZONTAL;
		constraints.anchor= GridBagConstraints.LINE_END;
		constraints.weightx= 0.7;
		textField2.setBorder( BorderFactory.createTitledBorder(
				BorderFactory.createEtchedBorder(EtchedBorder.RAISED, Color.GRAY,Color.DARK_GRAY), "#of iterations"
		));
		panel.add(textField2,constraints);

		final JButton clusterIt= new JButton("Cluster");
		clusterIt.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				thread.run( ()-> {
					int k, iterations;
					double fuzziness;
					try {
						k= Integer.parseInt(numberOfClusters.getText());
						iterations= Integer.parseInt(textField2.getText());
						fuzziness= Double.parseDouble(textField.getText());
					} catch ( NumberFormatException nfe ) {
						//throw nfe;
						fuzziness= Utils.DEFAULT_FUZZINESS;
						k= Utils.DEFAULT_NUMBER_OF_CLUSTERS;
						iterations= Utils.DEFAULT_ITERS;
					}
					selectDistanceMeasure();
					getWindowSize();
					log.info(String.format("k= %d, fuzziness= %f",k,fuzziness));
					fuzzyKMeansClustering(k,fuzziness,iterations);
				});
			}
		});
		GridBagConstraints c= new GridBagConstraints();
		c.gridx= 0;
		c.gridy= 3;
		c.gridwidth= 1;
		c.gridheight= 1;
		c.fill= GridBagConstraints.HORIZONTAL;
		c.anchor= GridBagConstraints.CENTER;
		c.weightx= 0.7;
		panel.add(clusterIt,c);

		return panel;
	}

    private void selectDistanceMeasure() {
	    for ( Map.Entry<String,DistanceMeasure> entry: str2distance.entrySet() ) {
	        String x= entry.getKey();
	        if ( str2button.get(x).isSelected() ) {
	            selectedDistance= entry.getValue();
	            log.info("Selected as distance: "+x);
	            return ;
            }
        }
        selectedDistance= new EuclideanDistance();
        log.info("Selected as distance: Euclidean");
    }

    public static void main(final String[] args) {
		try {
			final GLCMClusteringFrame frame= new GLCMClusteringFrame();
			frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
			frame.setVisible(true);
		}
		catch ( final Exception e ) {
			e.printStackTrace();
		}
	}

	public OpService getOps() {
		return ops;
	}

	public void setOps(final OpService ops) {
		this.ops = ops;
	}

	public LogService getLog() {
		return log;
	}

	public void setLog(final LogService log) {
		this.log = log;
	}

	public StatusService getStatus() {
		return status;
	}

	public void setStatus(final StatusService status) {
		this.status = status;
	}

	public CommandService getCommand() {
		return cmd;
	}

	public void setCommand(final CommandService command) {
		this.cmd = command;
	}

	public ThreadService getThread() {
		return thread;
	}

	public void setThread(final ThreadService thread) {
		this.thread = thread;
	}

	public UIService getUi() {
		return ui;
	}

	public void setUi(final UIService ui) {
		this.ui = ui;
	}

	public void setDisplay( final ImageDisplay display ) {
		this.display= display;
	}

	public ImageDisplay getDisplay() {
		return display;
	}

	public void setOverlayService(OverlayService overlayService) {
		this.overlayService = overlayService;
		/*
		RealRect r= overlayService.getSelectionBounds(display);
		Img<UnsignedByteType> im= ArrayImgs.unsignedBytes((int)(r.height),(int)(r.width),3);
		RandomAccess<UnsignedByteType> ra = im.randomAccess();
		RandomAccess<UnsignedByteType> rr = currentSelection.randomAccess();
		for ( int i= 0, x= (int)r.x; x < (int)(r.x+r.height); ++x, ++i )
			for ( int j= 0, y= (int)r.y; y < (int)(r.y+r.width); ++y, ++j ) {
		    	ra.setPosition(0,x);
				ra.setPosition(1,y);
				ra.setPosition(2,0);
				rr.setPosition(0,i);
				rr.setPosition(1,j);
				rr.setPosition(2,0);
				rr.get().set(ra.get());
			}
			img= currentSelection;
			*/
	}

	public DatasetService getDatasetService() {
		return datasetService;
	}

	public void setDatasetService(DatasetService datasetService) {
		this.datasetService = datasetService;
	}

	public RandomAccessibleInterval<UnsignedByteType> getImg() {
		return img;
	}

	public void setImg(RandomAccessibleInterval<UnsignedByteType> img) {
		this.img = img;
	}

    public void setSelectedRegion(Img<UnsignedByteType> selectedRegionImg) {
	    this.selectedRegion= selectedRegionImg;
    }
}
