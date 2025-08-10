#!/usr/bin/env python3
"""
Batch Process Interviews

This script processes all interview files in a directory using the disfluency processor,
cleaning the text and saving the results to organized subfolders.

The processed files are saved to:
- data/processed/fluent_interviews/full_fluent_interviews/ (complete interviews with disfluencies removed, files named with _fluent suffix)
- data/processed/fluent_interviews/interviewer_fluent_interviews/ (for future use)
- data/processed/fluent_interviews/interviewee_fluent_interviews/ (for future use)

Usage:
    python process_interview_directory.py [input_dir] [output_dir]

If no directories are provided, it will use the default data/raw and data/processed directories.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the disfluency processor
try:
    from processors.disfluency_processor.disfluency_processor import DisfluencyProcessor
    from processors.disfluency_processor.disfluency_processor import (
        get_file_type, read_text_file, read_docx_file, 
        write_text_file, write_docx_file
    )
except ImportError as e:
    logger.error(f"Failed to import disfluency processor: {e}")
    sys.exit(1)

def get_supported_extensions() -> List[str]:
    """Return a list of supported file extensions."""
    return ['.txt', '.docx']

def process_file(
    processor: DisfluencyProcessor,
    input_path: Path,
    output_path: Path
) -> bool:
    """Process a single file with the disfluency processor."""
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input file based on type
        input_type = get_file_type(input_path)
        logger.info(f"Processing {input_type.upper()} file: {input_path}")
        
        if input_type == 'text':
            text = read_text_file(input_path)
        else:  # docx
            text = read_docx_file(input_path)
        
        if not text.strip():
            logger.warning(f"Skipping empty file: {input_path}")
            return False
        
        # Process text
        clean_text = processor.process_text(text)
        
        # Write output based on output file extension
        output_type = get_file_type(output_path)
        if output_type == 'text':
            write_text_file(output_path, clean_text)
        else:  # docx
            write_docx_file(output_path, clean_text)
        
        logger.info(f"Successfully processed: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False

def main():
    # Set up argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description='Process multiple interview files with disfluency removal.')
    parser.add_argument('input_dir', nargs='?', default='data/raw',
                       help='Directory containing input files (default: data/raw)')
    parser.add_argument('output_dir', nargs='?', default='data/processed',
                       help='Directory to save processed files (default: data/processed)')
    parser.add_argument('--ext', nargs='+', default=get_supported_extensions(),
                       help=f'File extensions to process (default: {get_supported_extensions()})')
    parser.add_argument('--model', default=None,
                       help='Path to the model file (default: joint-disfluency-detector-and-parser/best_models/swbd_fisher_bert_Edev.0.9078.pt)')
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Set default model path if not provided
    model_path = args.model
    if model_path is None:
        model_path = project_root / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    
    # Validate input directory
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the processor
    try:
        logger.info(f"Initializing disfluency processor with model: {model_path}")
        processor = DisfluencyProcessor(str(model_path))
    except Exception as e:
        logger.error(f"Failed to initialize disfluency processor: {e}")
        sys.exit(1)
    
    # Find all matching files
    input_files = []
    for ext in args.ext:
        # Ensure extension starts with a dot
        if not ext.startswith('.'):
            ext = f'.{ext}'
        input_files.extend(input_dir.glob(f'*{ext}'))
    
    if not input_files:
        logger.warning(f"No files found in {input_dir} with extensions: {args.ext}")
        return
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process each file
    success_count = 0
    for i, input_file in enumerate(input_files, 1):
        # Skip directories
        if not input_file.is_file():
            continue
            
        # Determine output path - save to full_fluent_interviews subfolder
        rel_path = input_file.relative_to(input_dir)
        # Create the full_fluent_interviews subfolder path
        fluent_interviews_dir = output_dir / 'fluent_interviews' / 'full_fluent_interviews'
        # Use _fluent suffix instead of .cleaned
        base_name = rel_path.stem
        output_file = fluent_interviews_dir / f"{base_name}_fluent{rel_path.suffix}"
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing file {i}/{len(input_files)}: {input_file.name}")
        
        # Process the file
        if process_file(processor, input_file, output_file):
            success_count += 1
    
    # Print summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successfully processed {success_count} of {len(input_files)} files")
    logger.info(f"Full fluent interviews saved to: {output_dir / 'fluent_interviews' / 'full_fluent_interviews'}")
    logger.info(f"Other subfolders available:")
    logger.info(f"  - {output_dir / 'fluent_interviews' / 'interviewer_fluent_interviews'}")
    logger.info(f"  - {output_dir / 'fluent_interviews' / 'interviewee_fluent_interviews'}")

if __name__ == "__main__":
    main()
