import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class ImageModelBuilderTest {

	@Rule
	public TemporaryFolder models = new TemporaryFolder();

	@Test
	public void test() throws Exception {
		Path images = Paths.get("images");
		Path model = models.newFile().toPath();
		new ImageModelBuilder(1).build(images, model);
	}
}
