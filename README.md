# docscan

## Module Descriptions

`edges.py` is a module that uses the Canny edge detection algorithm to find the edges of an image. Has adjustable sliders that control the upper and lower thresholds used for detection, allowing the user to see how they affect detection sensitvity.

`scanner.py` is a module that functions as a document scanner, using edge and contour detection to find the document and perspective transforms to produce as clear a scan of the document as possible. Also supports OCR.

**Usage**: 

* `python3 scanner.py [filename]` produces a scan of the image found at `filename`, shows a low quality preview to the user, and writes a high quality scan to disk.

* `python3 scanner.py [filename] --ocr` does the same thing, but also prints any text recognized in the scan to the console.

**Dependencies**:

See `requirements.txt`.
