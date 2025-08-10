# Disfluency Processor Tests

This folder contains test files for the disfluency processor functionality.

## Test Files

### `test_disfluency.py`
**Purpose**: Tests the core disfluency detection functionality with a simple example.
**What it does**:
- Loads the BERT-based disfluency detection model
- Tests with a sentence containing obvious disfluencies ("yeah", "right", "yes")
- Shows the parse tree structure and identifies disfluent nodes
- Demonstrates how disfluencies are removed from text

**Usage**:
```bash
python3 test_disfluency.py
```

### `test_processor.py`
**Purpose**: Tests the full disfluency processor with speaker tags and formatting.
**What it does**:
- Tests the complete processor pipeline
- Handles speaker sections with [Interviewer] and [Interviewee] tags
- Shows sentence splitting and processing
- Demonstrates the full text processing workflow

**Usage**:
```bash
python3 test_processor.py
```

### `test_simple.py`
**Purpose**: Tests the processor with a very basic example to verify core functionality.
**What it does**:
- Tests with a simple sentence without punctuation
- Verifies that the model can identify and remove basic disfluencies
- Shows the before/after comparison

**Usage**:
```bash
python3 test_simple.py
```

### `test_punctuation.py`
**Purpose**: Tests punctuation preservation during disfluency removal.
**What it does**:
- Tests with text containing punctuation
- Verifies that punctuation is preserved when disfluencies are removed
- Shows multiple examples of punctuation handling

**Usage**:
```bash
python3 test_punctuation.py
```

## Running All Tests

To run all tests:
```bash
cd tests/disfluency_tests
python3 test_disfluency.py
python3 test_processor.py
python3 test_simple.py
python3 test_punctuation.py
```

## Expected Results

- **test_disfluency.py**: Should show disfluencies being identified and removed
- **test_processor.py**: Should show speaker sections being processed correctly
- **test_simple.py**: Should show basic disfluency removal working
- **test_punctuation.py**: Should show punctuation being preserved

## Dependencies

All tests require:
- The BERT model files in `../../../model/`
- The disfluency processor model in `../../../joint-disfluency-detector-and-parser/best_models/`
- PyTorch and other dependencies from the main project

## File Structure

```
tests/disfluency_tests/
├── README.md              # This documentation file
├── test_disfluency.py     # Core disfluency detection test
├── test_processor.py      # Full processor pipeline test
├── test_simple.py         # Basic functionality test
└── test_punctuation.py    # Punctuation preservation test
```
