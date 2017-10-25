import static org.hamcrest.CoreMatchers.equalTo;
import static org.junit.Assert.assertThat;

import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.junit.Test;

public class ImageCaptionLabelGeneratorTest {

	private final PathLabelGenerator generator = new ImageCaptionLabelGenerator("Trail");

	@Test
	public void testImageWithMatchingDescription() {
		assertThat(getLabelForPath("images/P1530829_trail.JPG"), equalTo(new Text("trail")));
	}

	@Test
	public void testImageWithoutMatchingDescription() {
		assertThat(getLabelForPath("images/P1500414_beach.JPG"), equalTo(new Text("not_trail")));
	}

	private Writable getLabelForPath(String path) {
		return generator.getLabelForPath(getClass().getResource(path).getPath());
	}
}
