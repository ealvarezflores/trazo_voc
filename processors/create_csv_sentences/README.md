# Text to CSV Sentences Processor

A powerful tool for converting text and Word documents into CSV format with one sentence per row, implementing the exact same logic as the Streamlit app for Voice of Customer Analysis.

## Features

- **📝 Exact Streamlit Logic**: Implements the same text processing pipeline as the Streamlit app
- **⏰ Timestamp Removal**: Removes timestamps like "[Name] 00:00:00" or "[inaudible 00:00:00]"
- **🔍 Smart Sentence Filtering**: Filters sentences by word count (min/max length)
- **📄 Multi-Format Support**: Handles both TXT and DOCX input files
- **📊 Batch Processing**: Process entire directories of files automatically
- **📈 Consolidated Output**: Creates individual and consolidated CSV files
- **🗑️ Discarded Tracking**: Tracks and reports discarded sentences with reasons

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Single File Processing

```bash
python3 text_to_csv_processor.py data/raw/interview.txt data/processed/sentences.csv 4 200
```

**Parameters:**
- `input_file`: Path to input text or docx file
- `output_csv`: Path for output CSV file
- `min_words`: Minimum words per sentence (default: 4)
- `max_words`: Maximum words per sentence (default: 200)

### Batch Processing

```bash
python3 batch_processor.py data/raw data/processed/csv_sentences 4 200
```

**Parameters:**
- `input_directory`: Directory containing input files
- `output_directory`: Directory for output CSV files
- `min_words`: Minimum words per sentence (default: 4)
- `max_words`: Maximum words per sentence (default: 200)

### Python API

```python
from text_to_csv_processor import process_file, process_text

# Process a single file
process_file(Path("input.txt"), Path("output.csv"), 4, 200)

# Process text directly
sentences, discarded = process_text("Your text here", 4, 200)
```

## Output Format

### Main CSV Output
The processor creates CSV files with the same format as the Streamlit app:

| Column | Description |
|--------|-------------|
| `Input` | The sentence text |

### Discarded Sentences CSV
For each processed file, a separate CSV tracks discarded sentences:

| Column | Description |
|--------|-------------|
| `Sentence` | The discarded sentence text |
| `Word Count` | Number of words in the sentence |
| `Reason` | Why it was discarded ("Too short" or "Too long") |

### Batch Processing Output
When using batch processing, you get:

1. **Individual CSV files**: One CSV per input file
2. **Consolidated CSV**: All sentences from all files with file source tracking
3. **Discarded files**: Separate CSV files for discarded sentences

## Processing Logic

The processor implements the exact same logic as the Streamlit app:

### 1. Timestamp Removal
```python
# Remove patterns like "[Name] 00:00:00" or "[inaudible 00:00:00]"
text = re.sub(r'\[[^\]]+\]\s*\d{2}:\d{2}:\d{2}|\[[^\]]+\d{2}:\d{2}:\d{2}\]', ' ', text)
# Remove any remaining timestamps
text = re.sub(r'\d{2}:\d{2}:\d{2}', ' ', text)
```

### 2. Sentence Tokenization
Uses NLTK's `sent_tokenize()` for accurate sentence splitting:
```python
sentences = sent_tokenize(text)
```

### 3. Word Count Filtering
Filters sentences based on word count:
```python
wc = len(word_tokenize(sentence))
if sentence_min_len <= wc <= sentence_max_len:
    filtered_sentences.append(sentence)
else:
    # Add to discarded with reason
```

### 4. CSV Output
Creates CSV with 'Input' column (same as Streamlit app):
```python
writer.writerow({'Input': sentence})
```

## Examples

### Input Text
```
[Interviewer] 00:00:00 Hello, thank you for joining us today. How are you doing?

[Interviewee] 00:00:05 There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry.
```

### Output CSV
```csv
Input
Hello, thank you for joining us today.
How are you doing?
There is something called shop drawings, which is like a technical drawing thing.
That you need to have some technical knowledge.
But this is something that I know in this industry from every part of the industry.
```

### Discarded CSV
```csv
Sentence,Word Count,Reason
"Yeah, right.",2,Too short
"Of course.",2,Too short
"Yes.",1,Too short
```

## File Structure

```
processors/create_csv_sentences/
├── __init__.py
├── text_to_csv_processor.py    # Main processor
├── batch_processor.py          # Batch processing
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Integration with Disfluency Processor

This processor can be used in conjunction with the disfluency processor:

1. **Extract sentences** using this CSV processor
2. **Remove disfluencies** using the disfluency processor
3. **Re-analyze** the cleaned text for final statistics

## Testing

Create a test file to verify functionality:

```python
from text_to_csv_processor import process_text

# Test text with timestamps and disfluencies
test_text = """[Interviewer] 00:00:00 Hello, thank you for joining us today. How are you doing?

[Interviewee] 00:00:05 There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge."""

sentences, discarded = process_text(test_text, 4, 200)
print(f"Valid sentences: {len(sentences)}")
print(f"Discarded sentences: {len(discarded)}")
```

## Dependencies

- `python-docx`: For reading DOCX files
- `nltk`: For sentence tokenization
- `pandas`: For data manipulation (batch processing)
- `chardet`: For encoding detection (batch processing)

## License

This processor is part of the Trazo VOC project and follows the same license terms.
