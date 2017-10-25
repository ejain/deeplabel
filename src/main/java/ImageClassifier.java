import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import com.beust.jcommander.ParameterException;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ImageClassifier {

	private final Logger log = LoggerFactory.getLogger(getClass());
	private final MultiLayerNetwork network;
	private final NativeImageLoader loader = new NativeImageLoader(100, 100, 3);
	private final DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
	private final int labelIndex = 1;

	public ImageClassifier(Path model) {
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

	public double classify(Path image) {
		return network.output(loadImage(image)).getDouble(labelIndex);
	}

	private INDArray loadImage(Path path) {
		try {
			INDArray image = loader.asMatrix(path.toFile());
			scaler.transform(image);
			return image;
		} catch (IOException e) {
			throw new RuntimeException("Couldn't load image from " + path, e);
		}
	}

	static class Options {

		@Parameter(names = { "--model", "-m" }, required = true)
		String model;

		@Parameter(names = { "--images", "-i"}, required = true)
		String images;
	}

	public static void main(String[] args) throws IOException {
		try {
			Options options = new Options();
			JCommander.newBuilder().addObject(options).build().parse(args);
			ImageClassifier classifier = new ImageClassifier(Paths.get(options.model));
			Files.walk(Paths.get(options.images)).filter(Files::isRegularFile).forEach(path -> {
				System.out.printf("%5.2f%% %s%n", classifier.classify(path) * 100, path);
			});
		} catch (ParameterException e) {
			e.getJCommander().usage();
			System.exit(-1);
		}
	}
}
