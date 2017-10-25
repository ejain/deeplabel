import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ImageModelEvaluator {

	private final Logger log = LoggerFactory.getLogger(getClass());
	private final MultiLayerNetwork network;
	private final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	private final int labelIndex = 1;

	public ImageModelEvaluator(Path model) {
		network = loadModel(model);
	}

	private MultiLayerNetwork loadModel(Path path) {
		log.info("Loading model from {}...", path);
		try {
			return ModelSerializer.restoreMultiLayerNetwork(path.toFile());
		} catch (IOException e) {
			throw new RuntimeException("Couldn't load model from " + path, e);
		}
	}

	private void evaluate(Path images, String label) throws IOException {
		FileSplit testingData = new FileSplit(images.toFile(), NativeImageLoader.ALLOWED_FORMATS);
		log.info("Evaluating model using {} samples...", testingData.length());
		ImageRecordReader records = new ImageRecordReader(100, 100, 3, new ImageCaptionLabelGenerator(label));
		records.initialize(testingData);
		DataSetIterator datasets = new RecordReaderDataSetIterator(records, 10_000, labelIndex, 2);
		scaler.fit(datasets);
		datasets.setPreProcessor(scaler);
		log.info(network.evaluate(datasets).stats(true));
		log.info("Labels: {}", records.getLabels());
	}

	static class Options {

		@Parameter(names = { "--model", "-m" }, required = true)
		String model;

		@Parameter(names = { "--images", "-i"}, required = true)
		String images;

		@Parameter(names = { "--label", "-l"}, required = true)
		String label;
	}

	public static void main(String[] args) throws IOException {
		try {
			Options options = new Options();
			JCommander.newBuilder().addObject(options).build().parse(args);
			new ImageModelEvaluator(Paths.get(options.model)).evaluate(Paths.get(options.images), options.label);
		} catch (ParameterException e) {
			e.getJCommander().usage();
			System.exit(-1);
		}
	}
}
