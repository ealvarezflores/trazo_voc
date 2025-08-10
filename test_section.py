#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent / "processors" / "disfluency_processor"))

from disfluency_processor import DisfluencyProcessor

def test_section():
    """Test the disfluency processor on a small section of the interview."""
    
    # Initialize processor
    model_path = Path(__file__).parent / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    processor = DisfluencyProcessor(str(model_path))
    
    # Test with a small section from the interview that contains obvious disfluencies
    test_text = """[Interviewee] Leonardo Cipriani
There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry."""
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    # Process the text
    clean_text = processor.process_text(test_text)
    
    print("Processed text:")
    print(clean_text)
    
    # Test with just the content part
    print("\n" + "="*80 + "\n")
    content_only = "There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry."
    
    print("Content only - Original:")
    print(content_only)
    print("\nContent only - Processed:")
    clean_content = processor._process_single_chunk(content_only)
    print(clean_content)

if __name__ == "__main__":
    test_section()
