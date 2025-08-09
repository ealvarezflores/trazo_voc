import os
import re
import docx
import spacy
from pathlib import Path
from tqdm import tqdm

class TranscriptPreprocessor:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load('en_core_web_sm')
        
    def read_docx(self, file_path):
        """Extract text from a Word document."""
        doc = docx.Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    def clean_text(self, text):
        """Basic text cleaning."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?\-]', ' ', text)
        # Fix multiple spaces
        text = re.sub(' +', ' ', text)
        return text.strip()
    
    def process_transcript(self, text):
        """Process transcript with spaCy for better text normalization."""
        doc = self.nlp(text)
        processed_sentences = []
        
        for sent in doc.sents:
            # Basic cleaning for each sentence
            cleaned = self.clean_text(sent.text)
            if cleaned:  # Skip empty sentences
                processed_sentences.append(cleaned)
                
        return '\n'.join(processed_sentences)

def process_interviews(input_dir, output_dir):
    """Process all Word documents in the input directory."""
    preprocessor = TranscriptPreprocessor()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .docx files
    docx_files = list(input_dir.glob('*.docx'))
    
    if not docx_files:
        print(f"No .docx files found in {input_dir}")
        return
    
    print(f"Found {len(docx_files)} documents to process...")
    
    for docx_file in tqdm(docx_files, desc="Processing documents"):
        try:
            # Read and process the document
            text = preprocessor.read_docx(docx_file)
            processed_text = preprocessor.process_transcript(text)
            
            # Save the processed text
            output_file = output_dir / f"processed_{docx_file.name.replace('.docx', '.txt')}"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(processed_text)
                
        except Exception as e:
            print(f"Error processing {docx_file.name}: {str(e)}")

if __name__ == "__main__":
    # Define paths
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / 'data' / 'raw'
    output_dir = base_dir / 'data' / 'processed'
    
    # Process the interviews
    process_interviews(input_dir, output_dir)
