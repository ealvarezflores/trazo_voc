#!/usr/bin/env python3
"""
Separate Speakers

This script takes a fluent interview file and separates the content by speaker type
(interviewer vs interviewee), saving each type to its respective directory.

Usage:
    python3 processors/separate_speakers.py <input_file>
    python3 processors/separate_speakers.py data/processed/fluent_interviews/full_fluent_interviews/input_fluent.txt
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from docx import Document

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def extract_speaker_type(speaker_tag: str) -> str:
    """
    Extract speaker type from speaker tag.
    
    Args:
        speaker_tag: Speaker tag like "[Interviewer] Emilio Alvarez Flores"
        
    Returns:
        Speaker type: "interviewer" or "interviewee"
    """
    if speaker_tag.startswith('[Interviewer]'):
        return 'interviewer'
    elif speaker_tag.startswith('[Interviewee]'):
        return 'interviewee'
    else:
        # Default to interviewee for unknown tags
        logger.warning(f"Unknown speaker tag format: {speaker_tag}, defaulting to interviewee")
        return 'interviewee'


def extract_speaker_name(speaker_tag: str) -> str:
    """
    Extract speaker name from speaker tag.
    
    Args:
        speaker_tag: Speaker tag like "[Interviewer] Emilio Alvarez Flores"
        
    Returns:
        Speaker name: "Emilio Alvarez Flores"
    """
    # Remove the brackets and speaker type
    name = re.sub(r'^\[Interview(?:er|ee)\]', '', speaker_tag).strip()
    return name


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
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()])
        else:
            # Read text file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def parse_fluent_file(file_path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse a fluent interview file and separate content by speaker type.
    
    Args:
        file_path: Path to the fluent interview file
        
    Returns:
        Dictionary with 'interviewer' and 'interviewee' keys, each containing
        list of (speaker_name, content) tuples
    """
    content = read_file_content(file_path)
    if not content:
        return {'interviewer': [], 'interviewee': []}
    
    # Initialize result structure
    speakers = {'interviewer': [], 'interviewee': []}
    
    # Split content into lines
    lines = content.split('\n')
    
    current_speaker_type = None
    current_speaker_name = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        
        # Check if this is a speaker tag
        if re.match(r'^\[Interview(?:er|ee)\][^\n\]]*$', line):
            # Save previous speaker's content if any
            if current_speaker_type and current_speaker_name and current_content:
                content_text = '\n'.join(current_content).strip()
                if content_text:
                    speakers[current_speaker_type].append((current_speaker_name, content_text))
            
            # Start new speaker
            current_speaker_type = extract_speaker_type(line)
            current_speaker_name = extract_speaker_name(line)
            current_content = []
            
        elif line and current_speaker_type:
            # This is content for the current speaker
            current_content.append(line)
    
    # Don't forget the last speaker
    if current_speaker_type and current_speaker_name and current_content:
        content_text = '\n'.join(current_content).strip()
        if content_text:
            speakers[current_speaker_type].append((current_speaker_name, content_text))
    
    return speakers


def save_speaker_content(speakers: Dict[str, List[Tuple[str, str]]], 
                        base_output_dir: Path, 
                        original_filename: str) -> None:
    """
    Save separated speaker content to appropriate directories.
    
    Args:
        speakers: Dictionary with 'interviewer' and 'interviewee' content
        base_output_dir: Base output directory
        original_filename: Original filename (without extension)
    """
    # Create output directories
    interviewer_dir = base_output_dir / 'fluent_interviews' / 'interviewer_fluent_interviews'
    interviewee_dir = base_output_dir / 'fluent_interviews' / 'interviewee_fluent_interviews'
    
    interviewer_dir.mkdir(parents=True, exist_ok=True)
    interviewee_dir.mkdir(parents=True, exist_ok=True)
    
    # Save interviewer content
    if speakers['interviewer']:
        interviewer_file = interviewer_dir / f"{original_filename}_interviewer.txt"
        with open(interviewer_file, 'w', encoding='utf-8') as f:
            for speaker_name, content in speakers['interviewer']:
                f.write(f"[{speaker_name}]\n{content}\n\n")
        logger.info(f"Saved interviewer content to: {interviewer_file}")
        logger.info(f"  - {len(speakers['interviewer'])} interviewer sections")
    else:
        logger.warning("No interviewer content found")
    
    # Save interviewee content
    if speakers['interviewee']:
        interviewee_file = interviewee_dir / f"{original_filename}_interviewee.txt"
        with open(interviewee_file, 'w', encoding='utf-8') as f:
            for speaker_name, content in speakers['interviewee']:
                f.write(f"[{speaker_name}]\n{content}\n\n")
        logger.info(f"Saved interviewee content to: {interviewee_file}")
        logger.info(f"  - {len(speakers['interviewee'])} interviewee sections")
    else:
        logger.warning("No interviewee content found")


def process_file(input_file: Path, output_dir: Path = None) -> bool:
    """
    Process a single fluent interview file.
    
    Args:
        input_file: Path to the fluent interview file
        output_dir: Output directory (defaults to input_file's parent's parent)
        
    Returns:
        True if successful, False otherwise
    """
    if not input_file.exists():
        logger.error(f"Input file does not exist: {input_file}")
        return False
    
    # Determine output directory
    if output_dir is None:
        # Go up two levels from the input file to get to the base processed directory
        output_dir = input_file.parent.parent.parent
    
    # Get original filename without extension
    original_filename = input_file.stem
    
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Parse the file
    speakers = parse_fluent_file(input_file)
    
    # Log statistics
    total_interviewer = len(speakers['interviewer'])
    total_interviewee = len(speakers['interviewee'])
    logger.info(f"Found {total_interviewer} interviewer sections and {total_interviewee} interviewee sections")
    
    # Save separated content
    save_speaker_content(speakers, output_dir, original_filename)
    
    return True


def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Separate fluent interview content by speaker type.')
    parser.add_argument('input_file', help='Path to the fluent interview file')
    parser.add_argument('--output-dir', help='Output directory (defaults to input file parent)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    success = process_file(input_file, output_dir)
    
    if success:
        logger.info("Processing completed successfully!")
    else:
        logger.error("Processing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
