#!/usr/bin/env python3
"""
Test script for the Text to CSV Processor
"""

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "processors" / "create_csv_sentences"))

from text_to_csv_processor import process_text, remove_timestamps, split_into_paragraphs


def test_timestamp_removal():
    """Test timestamp removal functionality."""
    print("Testing timestamp removal:")
    print("-" * 40)
    
    test_text = """[Interviewer] 00:00:00 Hello, thank you for joining us today.
[Interviewee] 00:00:05 There is something called shop drawings.
[inaudible 00:00:10] Yeah, right. That you need to have some technical knowledge."""
    
    cleaned_text = remove_timestamps(test_text)
    print("Original text:")
    print(test_text)
    print("\nCleaned text:")
    print(cleaned_text)
    print()


def test_sentence_splitting():
    """Test sentence splitting functionality."""
    print("Testing sentence splitting:")
    print("-" * 40)
    
    test_text = """Hello, thank you for joining us today. How are you doing? There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry."""
    
    filtered_sentences, discarded_sentences = split_into_paragraphs(test_text, 4, 200)
    
    print("Original text:")
    print(test_text)
    print(f"\nFiltered sentences ({len(filtered_sentences)}):")
    for i, sentence in enumerate(filtered_sentences, 1):
        print(f"{i}. {sentence}")
    
    print(f"\nDiscarded sentences ({len(discarded_sentences)}):")
    for discarded in discarded_sentences:
        print(f"- '{discarded['Sentence']}' (Word Count: {discarded['Word Count']}, Reason: {discarded['Reason']})")
    print()


def test_full_processing():
    """Test the complete processing pipeline."""
    print("Testing complete processing pipeline:")
    print("-" * 40)
    
    test_text = """[Interviewer] 00:00:00 Hello, thank you for joining us today. How are you doing?

[Interviewee] 00:00:05 There is something called shop drawings, which is like a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry.

[Interviewer] 00:00:15 That's very interesting. Um, could you tell us more about that?

[Interviewee] 00:00:20 Well, you know, it's like... it's a process where you take the design and you make it, you know, construction-ready. I mean, the contractors can actually build from it."""
    
    sentences, discarded = process_text(test_text, 4, 200)
    
    print("Original text:")
    print(test_text)
    print(f"\nValid sentences ({len(sentences)}):")
    for i, sentence in enumerate(sentences, 1):
        print(f"{i}. {sentence}")
    
    print(f"\nDiscarded sentences ({len(discarded)}):")
    for discarded_item in discarded:
        print(f"- '{discarded_item['Sentence']}' (Word Count: {discarded_item['Word Count']}, Reason: {discarded_item['Reason']})")
    print()


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("Testing edge cases:")
    print("-" * 40)
    
    # Test with very short sentences
    short_text = "Hi. Hello. How are you? I'm fine. Thanks."
    sentences, discarded = process_text(short_text, 3, 10)
    print(f"Short text test - Valid: {len(sentences)}, Discarded: {len(discarded)}")
    
    # Test with very long sentences
    long_text = "This is a very long sentence that contains many words and should probably be split into multiple smaller sentences for better readability and comprehension by the reader who might find it difficult to follow along with such a lengthy sentence that goes on and on without any breaks or pauses."
    sentences, discarded = process_text(long_text, 4, 20)
    print(f"Long text test - Valid: {len(sentences)}, Discarded: {len(discarded)}")
    
    # Test with empty text
    empty_text = ""
    sentences, discarded = process_text(empty_text, 4, 200)
    print(f"Empty text test - Valid: {len(sentences)}, Discarded: {len(discarded)}")
    
    # Test with only timestamps
    timestamp_text = "[Interviewer] 00:00:00 [Interviewee] 00:00:05 [inaudible 00:00:10]"
    sentences, discarded = process_text(timestamp_text, 4, 200)
    print(f"Timestamp-only test - Valid: {len(sentences)}, Discarded: {len(discarded)}")
    print()


if __name__ == "__main__":
    print("Text to CSV Processor Test Suite")
    print("=" * 50)
    print()
    
    test_timestamp_removal()
    test_sentence_splitting()
    test_full_processing()
    test_edge_cases()
    
    print("Test suite completed!")
