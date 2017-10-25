import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Objects;

import org.apache.tika.exception.TikaException;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.metadata.TikaCoreProperties;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.jpeg.JpegParser;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class ImageCaptionLabelGenerator implements PathLabelGenerator {

	private final Logger log = LoggerFactory.getLogger(getClass());
	private final String descriptionToMatch;
	private final Text labelForMatch, labelForNoMatch;

	public ImageCaptionLabelGenerator(String descriptionToMatch) {
		this.descriptionToMatch = descriptionToMatch;
		this.labelForMatch = new Text(descriptionToMatch.toLowerCase());
		this.labelForNoMatch = new Text("not_" + descriptionToMatch.toLowerCase());
	}

	@Override
	public Writable getLabelForPath(String path) {
		return isImageWithMatchingDescription(path) ? labelForMatch : labelForNoMatch;
	}

	private boolean isImageWithMatchingDescription(String path) {
		try (InputStream stream = new BufferedInputStream(new FileInputStream(path))) {
			Metadata metadata = new Metadata();
			new JpegParser().parse(stream, new DefaultHandler(), metadata, new ParseContext());
			String description = metadata.get(TikaCoreProperties.DESCRIPTION);
			return Objects.equals(description, descriptionToMatch);
		} catch (IOException|TikaException|SAXException e) {
			log.error("Couldn't extract a description from {}", path, e);
		}
		return false;
	}

	@Override
	public Writable getLabelForPath(URI uri) {
		return getLabelForPath(new File(uri).toString());
	}
}
