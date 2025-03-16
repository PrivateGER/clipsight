# CLIPsight

A graphical user interface application for searching images using CLIP embeddings. 
This tool allows you to search through your image collection using natural language descriptions or similar image queries.

<table>
  <tr>
    <td><img src="https://s3.plasmatrap.com/plasmatrap/2d552eaa-12dd-4ea0-b336-cf4c1ad76e25.webp" alt="Sample Text Search"><p style="text-align: center">Natural language search</p></td>
    <td><img src="https://s3.plasmatrap.com/plasmatrap/222769f3-33d9-4ad7-8960-2fa482418923.webp" alt="Sample Image Search"><p style="text-align: center">Image-based search</p></td>
  </tr>
</table>

## Features

- **Text-based Image Search**: Find images that match your text descriptions using CLIP's multimodal understanding
- **Image-based Search**: Find visually similar images by using an image as your query
- **Embedding Generation**: Create and manage embeddings for your image collections
- **Batch Processing**: Efficiently process large image collections with batched operations
- **Thumbnail Management**: Automatic thumbnail generation and caching for fast browsing
- **Adaptive Threshold**: Smart thresholding to show only relevant results
- **Cross-platform**: Works on Windows, macOS, and Linux
- **Light/Dark Theme Support**: Choose your preferred visual theme
- **GPU Acceleration**: CUDA support for significantly faster processing

![Embed Generation in GUI](https://s3.plasmatrap.com/plasmatrap/52159e20-8ea7-4103-9e98-313661e024de.png)

## Requirements

- Python 3.6+
- NVIDIA CUDA Toolkit 11.0+ (for GPU acceleration)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/PrivateGER/clipsight
   cd clipsight
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Launch the application:
   ```
   python main.py
   ```

## Usage

### Generating Embeddings

1. Switch to the "Generate Embeddings" tab
2. Select a directory containing images
3. Choose an output file for storing embeddings
4. Select a CLIP model (default: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`)
5. Click "Generate Embeddings"
6. Wait for the process to complete (progress will be displayed)

### Searching Images

1. In the "Search" tab, load your embeddings file
2. Enter a text query in the "Text Query" field or select an image for "Image Query"
3. Adjust the similarity threshold as needed
4. Browse through your results using the pagination controls
5. Click on any image to open it in your default image viewer
6. Right-click on an image for additional options like "Search Similar Images"

## Command Line Interface

You can also generate embeddings from the command line:

```
python generate.py --directory /path/to/images --output embeddings.json.zst --model laion/CLIP-ViT-H-14-laion2B-s32B-b79K
```

Optional arguments:
- `--cpu`: Force CPU inference even if GPU is available
- `--fp16`: Use half-precision for faster processing with lower memory usage (but also lower accuracy)
- `--batch-size`: Number of images to process at once (default: 16)
- `--save-interval`: Save progress after processing N batches (default: 10)

## License

This project is open source under AGPL. See the [LICENSE](LICENSE) file for more details.