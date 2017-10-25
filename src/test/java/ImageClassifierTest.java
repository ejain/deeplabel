import static org.hamcrest.Matchers.*;
import static org.junit.Assert.assertThat;

import java.net.URISyntaxException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

public class ImageClassifierTest {

	@Test
	public void test() {
		ImageClassifier classifier = new ImageClassifier(Paths.get("models/trails.zip"));
		assertThat(classifier.classify(getResource("images/P1530829_trail.JPG")), greaterThan(0.9));
		assertThat(classifier.classify(getResource("images/P1500414_beach.JPG")), lessThan(0.1));
	}

	private Path getResource(String path) {
		try {
			return Paths.get(getClass().getResource(path).toURI());
		} catch (URISyntaxException e) {
			throw new RuntimeException(e);
		}
	}
}
