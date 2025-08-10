#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent / "processors" / "disfluency_processor"))

from disfluency_processor import DisfluencyProcessor

def test_larger_section():
    """Test the disfluency processor on a larger section of the interview."""
    
    # Initialize processor
    model_path = Path(__file__).parent / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    processor = DisfluencyProcessor(str(model_path))
    
    # Test with a larger section that should trigger chunking
    test_text = """[Interviewee] Leonardo Cipriani
There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry. Everybody's desperate for that. So because no one has a company here that they can rely on, they try to do themselves. And then whenever you're trying to do which is my case I have seven interior designs, designers and architects around the world working for me developing those shop technical drawings. At the same time they are adapting themselves to do that. It's not that they were meant to do that. They were interior designers or architects.

[Interviewer] Gabriel Almeida
Yeah.

[Interviewee] Leonardo Cipriani
Because I paid them. Good. Because I paid them in dollars. They're working for me and they are doing. Not because they want to, they're just doing it for the money."""
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    # Test the chunking process directly
    print("Testing chunking process:")
    words = test_text.split()
    print(f"Total words: {len(words)}")
    
    # Simulate the chunking process
    max_words_per_chunk = 30
    chunks = []
    for i in range(0, len(words), max_words_per_chunk):
        chunk_words = words[i:i + max_words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        print(f"Chunk {len(chunks)}: {chunk_text[:50]}...")
    
    print("\n" + "="*80 + "\n")
    
    # Process each chunk individually
    print("Processing each chunk:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"Original: {chunk}")
        processed = processor._process_single_chunk(chunk)
        print(f"Processed: {processed}")
    
    print("\n" + "="*80 + "\n")
    
    # Process the full text
    print("Processing full text:")
    clean_text = processor.process_text(test_text)
    print(clean_text)

if __name__ == "__main__":
    test_larger_section()
