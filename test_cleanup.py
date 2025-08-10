#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent / "processors" / "disfluency_processor"))

from cleanup_empty_speakers import cleanup_empty_speakers

def test_cleanup():
    """Test the cleanup function with sample text containing empty speaker sections."""
    
    # Sample text with empty speaker sections (like what we saw in the output)
    sample_text = """[Interviewee] Leonardo Cipriani
There is something called shop drawings, which is a technical drawing thing. Yeah, right. That you need to have some technical knowledge. Of course. Yes. But this is something that I know in this industry from every part of the industry. Everybody's desperate for that. So because no one has a company here that they can rely on, they try to do And then whenever you're trying to do which is my case here, I have seven interior designs, designers and architects around the world working for me developing themselves to do that. It's not that they were meant to do that. They were interior designers or architects.

[Interviewer] Gabriel Almeida


[Interviewee] Leonardo Cipriani
Because I paid them. Good. Because I paid them in dollars. They're working for me and they are doing. Not because they want to, they're just doing it for the money.

[Interviewer] Gabriel Almeida


[Interviewee] Leonardo Cipriani
The thing is the designers and architects they have on this side of the. Is more like creativity, innovation, it has to look But our brain, we are not very. How can I explain that? Our brain doesn't really work like process, step by step, engineering, your guys So we fuck it up all the time.

[Interviewer] Gabriel Almeida


[Interviewee] Leonardo Cipriani
so I'm saying that to you.

[Interviewer] Emilio Alvarez Flores
huh, why is that? Is it just the attention to detail versus artistic flair or.

[Interviewee] Leonardo Cipriani
No, just try to imagine yourself as a designer. So you're a designer, you need to think about colors matching this with that, blah, So your brain works more into this side of the job. But there's another side of which is the boring part, which is the technical part. But designers, they keep doing shit that doesn't work. So it works in their mind. Putting whenever you try to build something, it doesn't work. And I work with the best interior designers in Florida.

[Interviewer] Gabriel Almeida


[Interviewee] Leonardo Cipriani
And they don't know. Whatever they're doing is completely wrong."""
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*80 + "\n")
    
    # Apply cleanup
    cleaned_text = cleanup_empty_speakers(sample_text)
    
    print("Cleaned text:")
    print(cleaned_text)
    
    # Count empty speaker sections in original vs cleaned
    original_empty = sample_text.count('[Interviewer] Gabriel Almeida\n\n')
    cleaned_empty = cleaned_text.count('[Interviewer] Gabriel Almeida\n\n')
    
    print(f"\n" + "="*80 + "\n")
    print(f"Empty speaker sections removed: {original_empty - cleaned_empty}")

if __name__ == "__main__":
    test_cleanup()
