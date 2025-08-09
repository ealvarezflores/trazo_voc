# Trazo VOC

A project for processing and analyzing disfluencies in speech transcripts using state-of-the-art natural language processing techniques.

## Project Structure

```
trazo_voc/
├── data/                    # Data directory
│   ├── raw/                 # Raw input files
│   └── processed/           # Processed output files
├── joint-disfluency-detector-and-parser/  # Submodule for disfluency detection
└── processors/              # Processing modules
    └── disfluency_processor/
        ├── processor.py     # Main processor script
        └── requirements.txt # Dependencies
```

## Prerequisites

- Python 3.8+
- PyTorch (with MPS support recommended for Apple Silicon)
- Other dependencies listed in `processors/disfluency_processor/requirements.txt`

## Setup

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/yourusername/trazo_voc.git
   cd trazo_voc
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r processors/disfluency_processor/requirements.txt
   ```

4. Build the chart_helper module:
   ```bash
   cd joint-disfluency-detector-and-parser/src
   python setup.py build_ext --inplace
   cd ../..
   ```

## Usage

1. Place your input text file in `data/raw/input.txt`

2. Run the processor:
   ```bash
   python -m processors.disfluency_processor.processor
   ```

3. View the processed output in `data/processed/output.txt`

## Example

Input (`data/raw/input.txt`):
```
we don't uh I mean a lot of states don't have capital punishment
```

Output (`data/processed/output.txt`):
```
(S (EDITED (S (UNK we) (UNK don't))) (INTJ (UNK uh)) (PRN (S (NP (UNK I)) (VP (UNK mean)))) (NP (UNK a) (UNK lot)) (UNK of) (NP (UNK states)) (UNK don't) (VP (UNK have) (NP (UNK capital) (UNK punishment))))
```

## Model Details

The project uses a pre-trained joint disfluency detection and constituency parsing model based on the paper "Jointly Predicting Predicates and Arguments in Neural Semantic Role Labeling" (He et al., 2018).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The disfluency detection model is based on the [joint-disfluency-detector-and-parser](https://github.com/edunov/transformer-disfluency-detector) project.
