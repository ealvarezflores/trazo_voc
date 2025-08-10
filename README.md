# Trazo VOC

A powerful tool for processing and cleaning speech transcripts by removing disfluencies, filler words, and other speech artifacts. Ideal for preparing interview transcripts, meeting notes, and other spoken content for analysis or publication.

## Features

- **Advanced Disfluency Detection**: Uses a state-of-the-art BERT-based model trained on the Penn Treebank-3 Switchboard corpus
- **Intelligent Tag-Based Removal**: Removes disfluencies tagged as `EDITED`, `PRN`, `UH`, and `INTJ` nodes
- **Speaker-Aware Processing**: Preserves speaker tags (e.g., [Interviewer], [Interviewee]) while cleaning content
- **Multi-Format Support**: Processes both TXT and DOCX file formats
- **Batch Processing**: Process entire directories of interview files automatically
- **Smart Cleanup**: Automatically removes empty speaker sections after disfluency removal
- **Progress Tracking**: Real-time progress updates for large files
- **Punctuation Preservation**: Maintains original punctuation in non-disfluent parts of sentences
- **Organized Output**: Saves processed files with `_fluent` suffix in structured directories

## Project Structure

```
trazo_voc/
├── data/                                    # Data directory
│   ├── raw/                                 # Raw input files
│   └── processed/                           # Processed output files
│       └── fluent_interviews/               # Organized output structure
│           ├── full_fluent_interviews/      # Complete interviews with disfluencies removed
│           ├── interviewer_fluent_interviews/ # Future: interviewer-only content
│           └── interviewee_fluent_interviews/ # Future: interviewee-only content
├── joint-disfluency-detector-and-parser/    # Disfluency detection model
│   ├── best_models/                         # Pre-trained model files
│   ├── src/                                 # Model source code
│   └── model/                               # BERT model files
├── processors/                              # Processing modules
│   ├── disfluency_processor/                # Core disfluency processing
│   │   ├── disfluency_processor.py          # Main processor script
│   │   ├── cleanup_empty_speakers.py        # Empty speaker section cleanup
│   │   └── requirements.txt                 # Python dependencies
│   └── process_interview_directory.py       # Batch processing script
├── tests/                                   # Test files
│   └── disfluency_tests/                    # Disfluency processing tests
└── venv/                                    # Virtual environment
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

4. **Test the installation**:
   ```bash
   # Run a simple test
   python3 tests/disfluency_tests/test_simple.py
   ```

## Usage

### Single File Processing

Process a single text file and save the output:
```bash
python3 processors/disfluency_processor/disfluency_processor.py input.txt output.txt
```

### Batch Processing (Recommended)

Process all files in a directory:
```bash
python3 processors/process_interview_directory.py data/raw data/processed
```

This will:
- Process all `.txt` and `.docx` files in `data/raw/`
- Save cleaned files to `data/processed/fluent_interviews/full_fluent_interviews/`
- Automatically remove empty speaker sections
- Use `_fluent` suffix for output files
- **Automatically separate speakers** into interviewer and interviewee files

### Command Line Options

#### Single File Processing
```bash
usage: disfluency_processor.py [-h] input_file [output_file]

Process transcript text to remove disfluencies.

positional arguments:
  input_file           Path to the input text file
  output_file          Path to save the processed output (default: input_file.cleaned.docx)

options:
  -h, --help           show this help message and exit
```

#### Batch Processing
```bash
usage: process_interview_directory.py [-h] [input_dir] [output_dir] [--ext EXTENSIONS] [--model MODEL_PATH]

Process multiple interview files with disfluency removal.

positional arguments:
  input_dir            Directory containing input files (default: data/raw)
  output_dir           Directory to save processed files (default: data/processed)
  --ext EXTENSIONS     File extensions to process (default: .txt .docx)
  --model MODEL_PATH   Path to the model file
```

### Examples

1. **Process a single file**:
   ```bash
   python3 processors/disfluency_processor/disfluency_processor.py data/raw/interview.txt data/processed/clean_interview.txt
   ```

2. **Process all files in a directory**:
   ```bash
   python3 processors/process_interview_directory.py data/raw data/processed
   ```

3. **Process only text files**:
   ```bash
   python3 processors/process_interview_directory.py data/raw data/processed --ext .txt
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

### Output Structure

Processed files are saved with the following structure:
- **File naming**: Original filename with `_fluent` suffix (e.g., `interview.txt` → `interview_fluent.txt`)
- **Directory structure**: 
  - `data/processed/fluent_interviews/full_fluent_interviews/` (complete interviews)
  - `data/processed/fluent_interviews/interviewer_fluent_interviews/` (interviewer-only content with `_interviewer` suffix)
  - `data/processed/fluent_interviews/interviewee_fluent_interviews/` (interviewee-only content with `_interviewee` suffix)
- **Empty speaker cleanup**: Automatically removes speaker sections that become empty after disfluency removal
- **Speaker separation**: Automatically separates content by speaker type for focused analysis

## Advanced Usage

### Processing Large Files
For large transcripts, the processor automatically handles chunking and shows progress updates:

```
Progress: 25% complete (25/100 sections)
Progress: 50% complete (50/100 sections)
Progress: 100% complete
```

### Disfluency Detection Types

The model identifies and removes several types of disfluencies:
- **`EDITED`**: Speech repairs (reparandum, interregnum, repair)
- **`INTJ`**: Interjections (e.g., "um", "uh", "yeah", "right", "yes")
- **`PRN`**: Parenthetical asides (e.g., "you know", "I mean")
- **`UH`**: Filled pauses and hesitation markers

### Testing

Run the test suite to verify functionality:
```bash
# Test basic disfluency detection
python3 tests/disfluency_tests/test_simple.py

# Test punctuation preservation
python3 tests/disfluency_tests/test_punctuation.py

# Test full processor pipeline
python3 tests/disfluency_tests/test_processor.py
```

### Integration with Other Tools
You can pipe the output to other command-line tools:

```bash
python3 processors/disfluency_processor/disfluency_processor.py input.txt - | grep -i "keyword"
```

## Troubleshooting

- **Memory Issues**: For very large files, you might need to increase Python's memory limits
- **Model Loading Errors**: Ensure you've run `make download-model` to download the required model files
- **Build Errors**: Make sure you have all build dependencies installed (gcc, python3-dev)
- **MPS Compatibility**: The processor automatically uses CPU mode to avoid MPS compatibility issues on Apple Silicon
- **Empty Output**: If processing results in empty files, check that the input text contains speaker tags in the expected format
- **BERT Model Path**: If you encounter BERT model loading issues, ensure the model files are in `joint-disfluency-detector-and-parser/model/`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Technical Details

### Model Architecture
- **Base Model**: BERT (Bidirectional Encoder Representations from Transformers)
- **Training Data**: Penn Treebank-3 Switchboard corpus
- **Architecture**: Constituency parsing with disfluency detection
- **Framework**: PyTorch

### Processing Pipeline
1. **Text Input**: Raw interview text with speaker tags
2. **Speaker Sectioning**: Splits text by speaker sections
3. **Disfluency Detection**: Parses text to identify disfluency nodes
4. **Content Extraction**: Removes disfluent segments while preserving fluent content
5. **Punctuation Preservation**: Maintains original punctuation in cleaned text
6. **Empty Speaker Cleanup**: Removes speaker sections that become empty
7. **Output**: Clean, fluent text with organized file structure

## Acknowledgments

- The disfluency detection model is based on the [joint-disfluency-detector-and-parser](https://github.com/edunov/transformer-disfluency-detector) project.
- Special thanks to the open source community for their contributions to natural language processing tools and libraries.
