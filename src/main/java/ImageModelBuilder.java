import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import com.google.common.collect.ImmutableList;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.model.AlexNet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ImageModelBuilder {

	private final Logger log = LoggerFactory.getLogger(getClass());

	private final int height = 100;
	private final int width = 100;
	private final int channels = 3;
	private final int numLabels = 2;
	private final int iterations = 1;
	private final int epochs;
	private final int batchSize = 16;
	private final double trainingRatio = 0.8;
	private final long seed = 42;
	private final Random rand = new Random(seed);

	private final PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
	private final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	private final List<ImageTransform> transforms = ImmutableList.of(
		new FlipImageTransform(1) // mirror horizontally
		// new WarpImageTransform(rand, 42)
		// new ColorConversionTransform(new Random(seed), org.bytedeco.javacpp.opencv_imgproc.COLOR_BGR2YCrCb)
	);

	public ImageModelBuilder(int epochs) {
		this.epochs = epochs;
	}

	public void build(Path imagesPath, Path modelPath) throws IOException {
		try {
			MultiLayerNetwork network = initNetwork();
			InputSplit[] data = loadData(imagesPath);
			train(network, data[0]);
			evaluate(network, data[1]);
			save(network, modelPath);
		} finally {
			UIServer.getInstance().stop();
		}
	}

	private MultiLayerNetwork initNetwork() {
		log.info("Building model...");
		AlexNet model = new AlexNet(numLabels, seed, iterations);
		model.setInputShape(new int[][] { { channels, width, height } });
		MultiLayerNetwork network = model.init();
		StatsStorage statsStorage = new InMemoryStatsStorage();
		network.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(iterations));
		UIServer.getInstance().attach(statsStorage);
		return network;
	}

	private InputSplit[] loadData(Path path) {
		log.info("Loading data from {}...", path);
		BalancedPathFilter pathFilter = new BalancedPathFilter(rand, labelMaker, 0);
		FileSplit fileSplit = new FileSplit(path.toFile(), NativeImageLoader.ALLOWED_FORMATS, rand);
		return fileSplit.sample(pathFilter, trainingRatio, 1 - trainingRatio);
	}

	private void train(MultiLayerNetwork network, InputSplit data) throws IOException {
		log.info("Training model on {} samples...", data.length());
		ImageRecordReader records = new ImageRecordReader(height, width, channels, labelMaker);
		records.initialize(data);
		train(network, records);
		for (ImageTransform transform : transforms) {
			log.info("Training model on data transformed with {}...", transform.getClass());
			records.initialize(data, transform);
			train(network, records);
		}
	}

	private void train(MultiLayerNetwork network, ImageRecordReader records) {
		DataSetIterator datasets = new RecordReaderDataSetIterator(records, batchSize, 1, numLabels);
		scaler.fit(datasets);
		datasets.setPreProcessor(scaler);
		network.fit(new MultipleEpochsIterator(epochs, datasets));
	}

	private void evaluate(MultiLayerNetwork network, InputSplit testingData) throws IOException {
		log.info("Evaluating model using {} samples...", testingData.length());
		ImageRecordReader records = new ImageRecordReader(height, width, channels, labelMaker);
		records.initialize(testingData);
		DataSetIterator datasets = new RecordReaderDataSetIterator(records, batchSize, 1, numLabels);
		scaler.fit(datasets);
		datasets.setPreProcessor(scaler);
		log.info(network.evaluate(datasets).stats(true));
		log.info("Labels: {}", records.getLabels());
	}

	private void save(MultiLayerNetwork network, Path path) throws IOException {
		log.info("Saving model to {}...", path);
		ModelSerializer.writeModel(network, path.toFile(), false);
	}

	static class Options {

		@Parameter(names = { "--model", "-m" }, required = true)
		String model;

		@Parameter(names = { "--images", "-i"}, required = true)
		String images;

		@Parameter(names = { "--epochs", "-e"})
		int epochs = 100;
	}

	public static void main(String[] args) throws IOException {
		try {
			Options options = new Options();
			JCommander.newBuilder().addObject(options).build().parse(args);
			new ImageModelBuilder(options.epochs).build(Paths.get(options.images), Paths.get(options.model));
		} catch (ParameterException e) {
			e.getJCommander().usage();
			System.exit(-1);
		}
	}
}
