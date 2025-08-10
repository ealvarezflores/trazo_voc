#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent / "processors" / "disfluency_processor"))

from disfluency_processor import DisfluencyProcessor

def test_punctuation():
    """Test punctuation preservation with disfluency removal."""
    
    # Initialize processor
    model_path = Path(__file__).parent / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    processor = DisfluencyProcessor(str(model_path))
    
    # Test with text that has punctuation
    test_text = "There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes."
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    # Process the text
    clean_text = processor._process_single_chunk(test_text)
    
    print("Processed text:")
    print(clean_text)
    
    # Test with a more complex example
    print("\n" + "="*80 + "\n")
    test_text2 = "I think, um, that we should, you know, go to the store. Yeah, right. And then, like, buy some food."
    
    print("Original text 2:")
    print(test_text2)
    print("\n" + "="*80 + "\n")
    
    clean_text2 = processor._process_single_chunk(test_text2)
    
    print("Processed text 2:")
    print(clean_text2)

if __name__ == "__main__":
    test_punctuation()
