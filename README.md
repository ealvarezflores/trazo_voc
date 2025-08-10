# Trazo VOC

A powerful tool for processing and cleaning speech transcripts by removing disfluencies, filler words, and other speech artifacts. Ideal for preparing interview transcripts, meeting notes, and other spoken content for analysis or publication.

## Features

- Removes disfluencies, filler words, and speech artifacts
- Preserves speaker tags (e.g., [Interviewer], [Interviewee])
- Processes both TXT and DOCX file formats
- Progress tracking for large files
- Handles long transcripts with chunked processing
- Maintains original formatting and speaker attribution

## Project Structure

```
trazo_voc/
├── data/                    # Data directory
│   ├── raw/                 # Raw input files
│   └── processed/           # Processed output files
├── joint-disfluency-detector-and-parser/  # Disfluency detection model
└── processors/              # Processing modules
    └── disfluency_processor/
        ├── disfluency_processor.py     # Main processor script
        └── requirements.txt            # Python dependencies
```

## Prerequisites

- Python 3.8+
- PyTorch (with MPS support for Apple Silicon)
- Other dependencies listed in `processors/disfluency_processor/requirements.txt`

## Quick Start

1. **Clone the repository**:
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/trazo_voc.git
   cd trazo_voc
   ```

2. **Set up the environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r processors/disfluency_processor/requirements.txt
   
   # Build the chart_helper module
   cd joint-disfluency-detector-and-parser/src
   python setup.py build_ext --inplace
   cd ../..
   ```

3. **Download the model**:
   ```bash
   make download-model
   ```

## Usage

### Basic Usage

Process a text file and save the output:
```bash
python -m processors.disfluency_processor.disfluency_processor input.txt output.txt
```

### Command Line Options

```
usage: disfluency_processor.py [-h] [--format {txt,docx}] [--debug] input_file [output_file]

Process transcript text to remove disfluencies.

positional arguments:
  input_file           Path to the input text file
  output_file          Path to save the processed output (default: input_file.cleaned.txt)

options:
  -h, --help           show this help message and exit
  --format {txt,docx}  Output format (default: txt)
  --debug              Enable debug logging
```

### Examples

1. **Process a text file**:
   ```bash
   python -m processors.disfluency_processor.disfluency_processor data/raw/interview.txt data/processed/clean_interview.txt
   ```

2. **Process and save as DOCX**:
   ```bash
   python -m processors.disfluency_processor.disfluency_processor data/raw/meeting_notes.txt --format docx
   ```

3. **Process with debug logging**:
   ```bash
   python -m processors.disfluency_processor.disfluency_processor input.txt --debug
   ```

## Input Format

The processor expects input text with speaker tags in square brackets, for example:

```
[Interviewer] So, um, let's talk about your experience. Uh, how long have you been in this field?
[Interviewee] Well, I've been, like, working in tech for about, uh, ten years now?
```

## Output

The processor will output clean text with disfluencies removed while preserving speaker tags:

```
[Interviewer] Let's talk about your experience. How long have you been in this field?
[Interviewee] I've been working in tech for about ten years now.
```

## Advanced Usage

### Processing Large Files
For large transcripts, the processor automatically handles chunking and shows progress updates:

```
Progress: 25% complete (25/100 sections)
Progress: 50% complete (50/100 sections)
Progress: 100% complete
```

### Integration with Other Tools
You can pipe the output to other command-line tools:

```bash
python -m processors.disfluency_processor.disfluency_processor input.txt - | grep -i "keyword"
```

## Troubleshooting

- **Memory Issues**: For very large files, you might need to increase Python's memory limits
- **Model Loading Errors**: Ensure you've run `make download-model` to download the required model files
- **Build Errors**: Make sure you have all build dependencies installed (gcc, python3-dev)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The disfluency detection model is based on the [joint-disfluency-detector-and-parser](https://github.com/edunov/transformer-disfluency-detector) project.
- Special thanks to the open source community for their contributions to natural language processing tools and libraries.
