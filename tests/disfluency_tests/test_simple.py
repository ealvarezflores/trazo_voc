#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "processors" / "disfluency_processor"))

from disfluency_processor import DisfluencyProcessor

def test_simple():
    """Test with a very simple example."""
    
    # Initialize processor
    model_path = Path(__file__).parent.parent.parent / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    processor = DisfluencyProcessor(str(model_path))
    
    # Test with a very simple sentence
    test_text = "There is something called shop drawings which is like a technical drawing thing yeah right that you need to have some technical knowledge of course yes"
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    # Process the text
    clean_text = processor._process_single_chunk(test_text)
    
    print("Processed text:")
    print(clean_text)

if __name__ == "__main__":
    test_simple()
