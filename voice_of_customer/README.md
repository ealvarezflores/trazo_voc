# Voice of Customer Processing

This project processes and preprocesses voice-of-customer interview transcripts stored in Word documents using machine learning techniques for text cleaning and enhancement.

## Project Structure

```
voice_of_customer/
├── data/                    # Data files
│   ├── raw/                 # Raw Word documents
│   └── processed/           # Processed transcripts
├── models/                  # Saved models and weights
├── notebooks/               # Jupyter notebooks for exploration
├── src/                     # Source code
│   ├── __init__.py
│   ├── preprocess.py        # Text preprocessing utilities
│   └── models.py           # ML models for text processing
├── tests/                   # Test files
├── .gitignore
├── requirements.txt         # Python dependencies
└── README.md
```

## Setup

1. Create and activate virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download required NLTK data and spaCy model:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   python -m spacy download en_core_web_sm
   ```

## Usage

1. Place your Word documents in `data/raw/`
2. Run the preprocessing pipeline:
   ```bash
   python src/preprocess.py
   ```

## Features

- Text extraction from Word documents
- Basic text cleaning (removing special characters, extra whitespace, etc.)
- Advanced text preprocessing with NLTK and spaCy
- Machine learning-based text enhancement
- Configurable preprocessing pipeline

---

## Light Editing with Joint Disfluency Parser (Off-the-shelf)

This project can integrate the joint disfluency detector and constituency parser by Paria Jamshid Lou et al. to remove EDITED spans with minimal paraphrasing.

Repo: https://github.com/pariajm/joint-disfluency-detector-and-parser

### One-time setup

1) Clone the upstream repo under `third_party/`:
```bash
mkdir -p third_party
git clone https://github.com/pariajm/joint-disfluency-detector-and-parser third_party/joint-disfluency-detector-and-parser
```

2) Download a pretrained model (recommended):
```bash
mkdir -p third_party/joint-disfluency-detector-and-parser/best_models
curl -L -o third_party/joint-disfluency-detector-and-parser/best_models/swbd_fisher_bert_Edev.0.9078.pt \
  https://github.com/pariajm/joint-disfluency-detector-and-parser/releases/download/naacl2019/swbd_fisher_bert_Edev.0.9078.pt
```

3) Download BERT resources expected by the repo (placed in `model/` inside the repo):
```bash
mkdir -p third_party/joint-disfluency-detector-and-parser/model
curl -L -o third_party/joint-disfluency-detector-and-parser/model/bert-base-uncased-vocab.txt \
  https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
curl -L -o third_party/joint-disfluency-detector-and-parser/model/bert-base-uncased.tar.gz \
  https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
( cd third_party/joint-disfluency-detector-and-parser/model && tar -xf bert-base-uncased.tar.gz )
```

### Run light editing on .docx files

```bash
# Activate venv and install deps if not yet done
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the light-edit CLI
python -m src.light_edit.run \
  --repo third_party/joint-disfluency-detector-and-parser \
  --model third_party/joint-disfluency-detector-and-parser/best_models/swbd_fisher_bert_Edev.0.9078.pt
```

This will:
- Read `.docx` files from `data/raw/`
- Use the joint parser to obtain trees with `EDITED` nodes
- Remove tokens under `EDITED` spans (keep repairs)
- Apply minimal punctuation normalization
- Save cleaned `.txt` files to `data/processed/`
