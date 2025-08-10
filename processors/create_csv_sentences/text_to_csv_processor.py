#!/usr/bin/env python3
"""
Text to CSV Processor

This module implements the exact text-to-CSV conversion logic from the Streamlit app,
including timestamp removal, sentence filtering, and the same processing pipeline.
"""

import csv
import re
import sys
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from docx import Document
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data (uncomment if needed)
# nltk.download('punkt')

# Fix SSL certificate issues for NLTK downloads
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def remove_timestamps(text: str) -> str:
    """
    Remove timestamps from text (exact implementation from Streamlit app).
    
    Args:
        text: Input text
        
    Returns:
        Text with timestamps removed
    """
    # Remove patterns like "[Name] 00:00:00" or "[inaudible 00:00:00]"
    text = re.sub(r'\[[^\]]+\]\s*\d{2}:\d{2}:\d{2}|\[[^\]]+\d{2}:\d{2}:\d{2}\]', ' ', text)
    # Remove any remaining timestamps
    text = re.sub(r'\d{2}:\d{2}:\d{2}', ' ', text)
    return text.strip()


def split_into_paragraphs(text: str, sentence_min_len: int = 8, sentence_max_len: int = 200) -> Tuple[List[str], List[Dict]]:
    """
    Split text into sentences with filtering (exact implementation from Streamlit app).
    
    Args:
        text: Input text
        sentence_min_len: Minimum words per sentence
        sentence_max_len: Maximum words per sentence
        
    Returns:
        Tuple of (filtered_sentences, discarded_sentences)
    """
    # Try NLTK sentence splitting first, fallback to regex if not available
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fallback to regex-based sentence splitting if NLTK data not available
        logger.warning("NLTK punkt data not available, using regex fallback")
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
    
    filtered_sentences = []
    discarded_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip().replace('\n', ' ')
        # Use simple word splitting instead of NLTK word_tokenize
        wc = len(sentence.split())
        
        if sentence_min_len <= wc <= sentence_max_len:
            filtered_sentences.append(sentence)
        else:
            reason = "Too short" if wc < sentence_min_len else "Too long"
            discarded_sentences.append({
                "Sentence": sentence,
                "Word Count": wc,
                "Reason": reason
            })
    
    return filtered_sentences, discarded_sentences


def process_text_document(content: str, sentence_min_len: int, sentence_max_len: int) -> Tuple[List[str], List[Dict]]:
    """
    Process text content and return filtered and discarded sentences.
    
    Args:
        content: Input text content
        sentence_min_len: Minimum words per sentence
        sentence_max_len: Maximum words per sentence
        
    Returns:
        Tuple of (filtered_sentences, discarded_sentences)
    """
    content = remove_timestamps(content)
    return split_into_paragraphs(content, sentence_min_len, sentence_max_len)


def read_file_content(file_path: Path) -> str:
    """
    Read content from a file, handling both TXT and DOCX formats.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File content as string
    """
    try:
        if file_path.suffix.lower() == '.docx':
            # Read DOCX file
            doc = Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        else:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def write_csv(sentences: List[str], output_path: Path) -> bool:
    """
    Write sentences to CSV file in Streamlit app format.
    
    Args:
        sentences: List of sentences
        output_path: Output CSV file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Use the same column name as Streamlit app
            fieldnames = ['Input']  # Streamlit app uses 'Input' as column name
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for sentence in sentences:
                writer.writerow({
                    'Input': sentence
                })
        
        logger.info(f"Successfully wrote {len(sentences)} sentences to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing CSV file {output_path}: {e}")
        return False


def write_discarded_csv(discarded_sentences: List[Dict], output_path: Path) -> bool:
    """
    Write discarded sentences to CSV file.
    
    Args:
        discarded_sentences: List of discarded sentence dictionaries
        output_path: Output CSV file path
        
    Returns:
        True if successful, False otherwise
    """
    if not discarded_sentences:
        return True
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Sentence', 'Word Count', 'Reason']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for discarded in discarded_sentences:
                writer.writerow(discarded)
        
        logger.info(f"Successfully wrote {len(discarded_sentences)} discarded sentences to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error writing discarded CSV file {output_path}: {e}")
        return False


def process_file(input_path: Path, output_path: Path, sentence_min_len: int = 4, sentence_max_len: int = 200, 
                discarded_output_path: Optional[Path] = None) -> bool:
    """
    Process a single file from text to CSV (Streamlit app style).
    
    Args:
        input_path: Input file path
        output_path: Output CSV file path
        sentence_min_len: Minimum words per sentence
        sentence_max_len: Maximum words per sentence
        discarded_output_path: Optional path for discarded sentences CSV
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"Processing file: {input_path}")
        
        # Read file content
        text = read_file_content(input_path)
        if not text.strip():
            logger.warning(f"Empty file: {input_path}")
            return False
        
        # Process text document
        filtered_sentences, discarded_sentences = process_text_document(text, sentence_min_len, sentence_max_len)
        
        if not filtered_sentences:
            logger.warning(f"No valid sentences extracted from: {input_path}")
            return False
        
        # Write main CSV
        success = write_csv(filtered_sentences, output_path)
        
        # Write discarded sentences CSV if requested
        if discarded_sentences and discarded_output_path:
            write_discarded_csv(discarded_sentences, discarded_output_path)
        
        if success:
            logger.info(f"Successfully processed {len(filtered_sentences)} sentences from {input_path}")
            if discarded_sentences:
                logger.info(f"Discarded {len(discarded_sentences)} sentences")
        
        return success
        
    except Exception as e:
        logger.error(f"Error processing file {input_path}: {e}")
        return False


def process_text(text: str, sentence_min_len: int = 4, sentence_max_len: int = 200) -> Tuple[List[str], List[Dict]]:
    """
    Process text string directly.
    
    Args:
        text: Input text
        sentence_min_len: Minimum words per sentence
        sentence_max_len: Maximum words per sentence
        
    Returns:
        Tuple of (filtered_sentences, discarded_sentences)
    """
    return process_text_document(text, sentence_min_len, sentence_max_len)


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python text_to_csv_processor.py <input_file> <output_csv> [min_words] [max_words]")
        print("Example: python text_to_csv_processor.py data/raw/interview.txt data/processed/sentences.csv 4 200")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    # Optional parameters
    sentence_min_len = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    sentence_max_len = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)
    
    # Create discarded sentences output path
    discarded_output_path = output_path.with_stem(f"{output_path.stem}_discarded")
    
    # Process file
    success = process_file(input_path, output_path, sentence_min_len, sentence_max_len, discarded_output_path)
    
    if success:
        print(f"Successfully converted {input_path} to {output_path}")
        if discarded_output_path.exists():
            print(f"Discarded sentences saved to {discarded_output_path}")
    else:
        print(f"Error processing {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
