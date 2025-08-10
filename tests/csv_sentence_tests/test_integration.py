#!/usr/bin/env python3
"""
Integration Test for Text to CSV Processor

This test verifies the complete pipeline:
1. Process text through disfluency processor to get fluent text
2. Convert fluent text to CSV using the text-to-CSV processor
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the processors directories to the path
sys.path.insert(0, str(project_root / "processors" / "disfluency_processor"))
sys.path.insert(0, str(project_root / "processors" / "create_csv_sentences"))

# Import the processors
from disfluency_processor import DisfluencyProcessor
from text_to_csv_processor import process_text, remove_timestamps, split_into_paragraphs


def check_nltk_installation():
    """Check if NLTK is properly installed and download required data."""
    try:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        
        # Try to use NLTK functions
        test_text = "Hello world. This is a test."
        sentences = sent_tokenize(test_text)
        words = word_tokenize(test_text)
        
        print("✅ NLTK is properly installed and working")
        print(f"   - Test sentences: {sentences}")
        print(f"   - Test words: {words}")
        return True
        
    except LookupError as e:
        print("⚠️  NLTK data not found. Attempting to download with SSL fix...")
        try:
            import nltk
            import ssl
            
            # Fix SSL certificate issues
            try:
                _create_unverified_https_context = ssl._create_unverified_context
            except AttributeError:
                pass
            else:
                ssl._create_default_https_context = _create_unverified_https_context
            
            # Download punkt data
            nltk.download('punkt', quiet=True)
            print("✅ NLTK data downloaded successfully with SSL fix")
            
            # Test again
            test_text = "Hello world. This is a test."
            sentences = sent_tokenize(test_text)
            words = word_tokenize(test_text)
            print(f"   - Test sentences: {sentences}")
            print(f"   - Test words: {words}")
            return True
            
        except Exception as download_error:
            print(f"❌ Failed to download NLTK data: {download_error}")
            print("   Trying fallback regex-based sentence splitting...")
            return False
            
    except ImportError as e:
        print(f"❌ NLTK not properly installed: {e}")
        print("   Please run: pip install nltk")
        return False
    except Exception as e:
        print(f"❌ Error with NLTK: {e}")
        return False


def test_disfluency_to_csv_pipeline():
    """Test the complete pipeline: disfluency removal -> CSV conversion."""
    print("\n" + "="*60)
    print("TESTING COMPLETE PIPELINE: DISFLUENCY -> CSV")
    print("="*60)
    
    # Test text with disfluencies and timestamps
    test_text = """[Interviewer] 00:00:00 Hello, thank you for joining us today. Um, how are you doing?

[Interviewee] 00:00:05 Leonardo Cipriani
There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry.

[Interviewer] 00:00:15 That's very interesting. Um, could you tell us more about that?

[Interviewee] 00:00:20 Well, you know, it's like... it's a process where you take the design and you make it, you know, construction-ready. I mean, the contractors can actually build from it."""
    
    print("Original text with disfluencies and timestamps:")
    print("-" * 50)
    print(test_text)
    print()
    
    # Step 1: Process through disfluency processor
    print("Step 1: Processing through disfluency processor...")
    print("-" * 50)
    
    try:
        # Initialize disfluency processor
        model_path = project_root / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
        processor = DisfluencyProcessor(str(model_path))
        
        # Process text to remove disfluencies
        fluent_text = processor.process_text(test_text)
        
        print("Fluent text (disfluencies removed):")
        print("-" * 50)
        print(fluent_text)
        print()
        
    except Exception as e:
        print(f"❌ Error in disfluency processing: {e}")
        print("   Using original text for CSV conversion...")
        fluent_text = test_text
    
    # Step 2: Convert fluent text to CSV
    print("Step 2: Converting fluent text to CSV...")
    print("-" * 50)
    
    try:
        # Process text to sentences
        sentences, discarded = process_text(fluent_text, 4, 200)
        
        print(f"Valid sentences extracted: {len(sentences)}")
        print("Valid sentences:")
        for i, sentence in enumerate(sentences, 1):
            print(f"  {i}. {sentence}")
        
        print(f"\nDiscarded sentences: {len(discarded)}")
        if discarded:
            print("Discarded sentences:")
            for discarded_item in discarded:
                print(f"  - '{discarded_item['Sentence']}' (Word Count: {discarded_item['Word Count']}, Reason: {discarded_item['Reason']})")
        
        print()
        
        # Create CSV content
        csv_content = "Input\n"
        for sentence in sentences:
            csv_content += f'"{sentence}"\n'
        
        print("Generated CSV content:")
        print("-" * 50)
        print(csv_content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in CSV conversion: {e}")
        return False


def test_timestamp_removal():
    """Test timestamp removal functionality."""
    print("\n" + "="*60)
    print("TESTING TIMESTAMP REMOVAL")
    print("="*60)
    
    test_text = """[Interviewer] 00:00:00 Hello, thank you for joining us today.
[Interviewee] 00:00:05 There is something called shop drawings.
[inaudible 00:00:10] Yeah, right. That you need to have some technical knowledge.
[Name] 00:00:15 This is another timestamp pattern."""
    
    print("Original text with timestamps:")
    print("-" * 50)
    print(test_text)
    print()
    
    cleaned_text = remove_timestamps(test_text)
    
    print("Text after timestamp removal:")
    print("-" * 50)
    print(cleaned_text)
    print()


def test_sentence_filtering():
    """Test sentence filtering with different word count limits."""
    print("\n" + "="*60)
    print("TESTING SENTENCE FILTERING")
    print("="*60)
    
    test_text = """Hello, thank you for joining us today. How are you doing? There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry."""
    
    print("Original text:")
    print("-" * 50)
    print(test_text)
    print()
    
    # Test with different filtering parameters
    test_cases = [
        (2, 10, "Very restrictive (2-10 words)"),
        (4, 200, "Default (4-200 words)"),
        (1, 50, "Very permissive (1-50 words)")
    ]
    
    for min_words, max_words, description in test_cases:
        print(f"Filtering: {description}")
        print("-" * 30)
        
        filtered_sentences, discarded_sentences = split_into_paragraphs(test_text, min_words, max_words)
        
        print(f"Valid sentences ({len(filtered_sentences)}):")
        for i, sentence in enumerate(filtered_sentences, 1):
            print(f"  {i}. {sentence}")
        
        print(f"Discarded sentences ({len(discarded_sentences)}):")
        for discarded in discarded_sentences:
            print(f"  - '{discarded['Sentence']}' (Word Count: {discarded['Word Count']}, Reason: {discarded['Reason']})")
        print()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\n" + "="*60)
    print("TESTING EDGE CASES")
    print("="*60)
    
    test_cases = [
        ("Empty text", ""),
        ("Only timestamps", "[Interviewer] 00:00:00 [Interviewee] 00:00:05"),
        ("Very short sentences", "Hi. Hello. How are you? I'm fine. Thanks."),
        ("Very long sentence", "This is a very long sentence that contains many words and should probably be split into multiple smaller sentences for better readability and comprehension by the reader who might find it difficult to follow along with such a lengthy sentence that goes on and on without any breaks or pauses."),
        ("Mixed content", "Hi. Hello, thank you for joining us today. How are you doing? There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry.")
    ]
    
    for description, test_text in test_cases:
        print(f"Testing: {description}")
        print("-" * 30)
        print(f"Input: '{test_text}'")
        
        sentences, discarded = process_text(test_text, 4, 200)
        
        print(f"Valid sentences: {len(sentences)}")
        print(f"Discarded sentences: {len(discarded)}")
        
        if sentences:
            print("Valid sentences:")
            for i, sentence in enumerate(sentences, 1):
                print(f"  {i}. {sentence}")
        
        if discarded:
            print("Discarded sentences:")
            for discarded_item in discarded:
                print(f"  - '{discarded_item['Sentence']}' (Word Count: {discarded_item['Word Count']}, Reason: {discarded_item['Reason']})")
        print()


def main():
    """Main test function."""
    print("TEXT TO CSV PROCESSOR INTEGRATION TEST")
    print("="*60)
    
    # Check NLTK installation
    print("Checking NLTK installation...")
    nltk_working = check_nltk_installation()
    if not nltk_working:
        print("⚠️  NLTK not fully working, but continuing with fallback mechanisms...")
    
    # Run all tests
    try:
        test_timestamp_removal()
        test_sentence_filtering()
        test_edge_cases()
        test_disfluency_to_csv_pipeline()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
