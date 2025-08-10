#!/usr/bin/env python3
"""
Cleanup Empty Speakers

This module removes empty speaker sections from processed interview text.
When disfluencies are removed, sometimes entire speaker sections become empty,
leaving just the speaker tag with no content. This module identifies and removes
such empty sections.
"""

import re
from pathlib import Path
from typing import List, Tuple


def cleanup_empty_speakers(text: str) -> str:
    """
    Remove empty speaker sections from processed interview text.
    
    Args:
        text: The processed interview text with speaker tags
        
    Returns:
        Cleaned text with empty speaker sections removed
    """
    if not text.strip():
        return text
    
    # Split text into lines
    lines = text.split('\n')
    cleaned_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Check if this line is a speaker tag
        if re.match(r'^\[Interview(?:er|ee)\][^\n\]]*$', line):
            speaker_tag = line
            
            # Look ahead to see if there's any content after this speaker tag
            content_found = False
            
            # Check the next few lines for content or another speaker tag
            j = i + 1
            while j < len(lines) and j < i + 10:  # Look ahead up to 10 lines
                next_line = lines[j].strip()
                
                # If we find another speaker tag, stop looking
                if re.match(r'^\[Interview(?:er|ee)\][^\n\]]*$', next_line):
                    break
                
                # If we find non-empty content, mark it
                if next_line and not next_line.startswith('['):
                    content_found = True
                    break
                
                j += 1
            
            # If content was found, keep the speaker tag
            if content_found:
                cleaned_lines.append(line)
            # If no content found, skip this speaker tag (don't add it to cleaned_lines)
            
        else:
            # Not a speaker tag, keep the line
            cleaned_lines.append(line)
        
        i += 1
    
    # Join the lines back together
    result = '\n'.join(cleaned_lines)
    
    # Clean up multiple consecutive empty lines
    result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
    
    return result.strip()


def process_file(input_path: Path, output_path: Path) -> bool:
    """
    Process a single file to remove empty speaker sections.
    
    Args:
        input_path: Path to the input file
        output_path: Path to save the cleaned file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read the input file
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Clean up empty speakers
        cleaned_content = cleanup_empty_speakers(content)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the cleaned content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def main():
    """Main function for command line usage."""
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python cleanup_empty_speakers.py <input_file> <output_file>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        print(f"Input file does not exist: {input_path}")
        sys.exit(1)
    
    success = process_file(input_path, output_path)
    if success:
        print(f"Successfully cleaned {input_path} -> {output_path}")
    else:
        print(f"Failed to process {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
