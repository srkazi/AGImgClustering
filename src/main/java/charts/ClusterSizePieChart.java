package charts;

import javax.swing.*;

import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PiePlot;
import org.jfree.data.general.DefaultPieDataset;
import org.jfree.data.general.PieDataset;

import java.awt.*;
import java.util.Map;

import static org.jfree.chart.ChartFactory.createPieChart;

public class ClusterSizePieChart extends JFrame {

    private Map<String,Object[]> map;

    public ClusterSizePieChart(String title, Map<String,Object[]> map ) {
        super( title );
        this.map= map;
        setContentPane(createDemoPanel( ));
    }

    private PieDataset createDataset() {
        DefaultPieDataset dataset = new DefaultPieDataset();
        for ( Map.Entry<String,Object[]> entry: map.entrySet() )
            dataset.setValue(entry.getKey(), (Double)(entry.getValue()[1]) );
        return dataset;
    }

    private JFreeChart createChart( PieDataset dataset ) {
        JFreeChart chart = createPieChart(
                "Cluster Sizes",   // chart title
                dataset,          // data
                true,             // include legend
                true,
                false);
        PiePlot plot= (PiePlot)chart.getPlot();
        for ( Map.Entry<String,Object[]> entry: map.entrySet() ) {
            Color color= (Color)(entry.getValue()[0]);
            plot.setSectionPaint(entry.getKey(),color);
        }
        return chart;
    }

    public JPanel createDemoPanel( ) {
        JFreeChart chart = createChart(this.createDataset( ) );
        return new ChartPanel( chart );
    }
}
