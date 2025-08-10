#!/usr/bin/env python3
"""
Batch Process Interviews

This script processes all interview files in a directory using the disfluency processor,
cleaning the text and saving the results to organized subfolders.

The processed files are saved to:
- data/processed/fluent_interviews/full_fluent_interviews/ (complete interviews with disfluencies removed and empty speaker sections cleaned, files named with _fluent suffix)
- data/processed/fluent_interviews/interviewer_fluent_interviews/ (interviewer-only content, files named with _interviewer suffix)
- data/processed/fluent_interviews/interviewee_fluent_interviews/ (interviewee-only content, files named with _interviewee suffix)
- data/processed/fluent_interviewer_sentence_csv/ (CSV files from interviewer content)
- data/processed/fluent_interviewee_sentence_csv/ (CSV files from interviewee content)

Usage:
    python process_interview_directory.py [input_dir] [output_dir] [--process-interviewer-to-csv]

If no directories are provided, it will use the default data/raw and data/processed directories.
"""

import os
import sys
import logging
import re
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import List, Optional, Tuple
from functools import partial

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
    from processors.disfluency_processor.cleanup_empty_speakers import cleanup_empty_speakers
    from processors.separate_speakers import process_file as separate_speakers
    from processors.create_csv_sentences.text_to_csv_processor import process_file as process_csv_file
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def get_supported_extensions() -> List[str]:
    """Return a list of supported file extensions."""
    return ['.txt', '.docx']

def strip_speaker_tags(text: str) -> str:
    """
    Remove speaker tags like [Interviewer] and [Interviewee] from the text.
    
    Args:
        text: Input text with speaker tags
        
    Returns:
        Text with speaker tags removed
    """
    # Remove speaker tags and any leading/trailing whitespace
    text = re.sub(r'^\[Interview(?:er|ee)\][^\n]*\n?', '', text, flags=re.MULTILINE)
    # Clean up any remaining speaker tags in the middle of text
    text = re.sub(r'\n\[Interview(?:er|ee)\][^\n]*\n?', '\n', text)
    # Clean up multiple newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def determine_speaker_type(file_path: Path) -> str:
    """
    Determine if a file contains interviewer or interviewee content based on the filename.
    
    Args:
        file_path: Path to the fluent interview file
        
    Returns:
        'interviewer' or 'interviewee' based on filename
    """
    filename = file_path.name.lower()
    if '_interviewer' in filename:
        return 'interviewer'
    elif '_interviewee' in filename:
        return 'interviewee'
    else:
        # Default to interviewee if we can't determine
        logger.warning(f"Could not determine speaker type from filename: {file_path.name}. Defaulting to interviewee.")
        return 'interviewee'

def process_csv_for_speaker(
    fluent_file_path: Path,
    output_dir: Path,
    speaker_type: str,
    process_interviewer: bool = False
) -> bool:
    """
    Process a fluent interview file to CSV based on speaker type.
    
    Args:
        fluent_file_path: Path to the fluent interview file
        output_dir: Base output directory
        speaker_type: 'interviewer' or 'interviewee'
        process_interviewer: Whether to process interviewer files
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Skip interviewer files unless explicitly requested
        if speaker_type == 'interviewer' and not process_interviewer:
            logger.info(f"Skipping interviewer file (use --process-interviewer-to-csv to process): {fluent_file_path.name}")
            return True
        
        # For interviewer files, we would run another processor first
        # For now, we'll just log this and continue
        if speaker_type == 'interviewer':
            logger.info(f"Note: Interviewer file {fluent_file_path.name} would run through another processor first")
            # TODO: Add interviewer-specific processor here
        
        # Read the fluent file content
        file_type = get_file_type(fluent_file_path)
        if file_type == 'text':
            content = read_text_file(fluent_file_path)
        else:  # docx
            content = read_docx_file(fluent_file_path)
        
        if not content.strip():
            logger.warning(f"Empty fluent file: {fluent_file_path}")
            return False
        
        # Strip speaker tags before processing
        clean_content = strip_speaker_tags(content)
        
        if not clean_content.strip():
            logger.warning(f"No content remaining after stripping speaker tags: {fluent_file_path}")
            return False
        
        # Determine output CSV path with subfolders
        csv_dir = output_dir / f"fluent_{speaker_type}_sentence_csv"
        validated_dir = csv_dir / "validated_sentences"
        discarded_dir = csv_dir / "discarded_sentences"
        
        # Create directories
        validated_dir.mkdir(parents=True, exist_ok=True)
        discarded_dir.mkdir(parents=True, exist_ok=True)
        
        # Create CSV filename
        base_name = fluent_file_path.stem
        # Remove the _fluent suffix if present
        if base_name.endswith('_fluent'):
            base_name = base_name[:-7]  # Remove '_fluent'
        csv_filename = f"{base_name}_{speaker_type}_sentences.csv"
        csv_path = validated_dir / csv_filename
        
        # Create a temporary file with stripped content for CSV processing
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as temp_file:
            temp_file.write(clean_content)
            temp_file_path = Path(temp_file.name)
        
        try:
            # Create discarded sentences output path in the discarded_sentences subfolder
            discarded_filename = f"{base_name}_{speaker_type}_sentences_discarded.csv"
            discarded_path = discarded_dir / discarded_filename
            
            # Process to CSV using the text_to_csv_processor
            success = process_csv_file(
                input_path=temp_file_path,
                output_path=csv_path,
                sentence_min_len=4,
                sentence_max_len=200,
                discarded_output_path=discarded_path
            )
        finally:
            # Clean up temporary file
            if temp_file_path.exists():
                temp_file_path.unlink()
        
        if success:
            logger.info(f"Successfully created CSV for {speaker_type}: {csv_path}")
        else:
            logger.error(f"Failed to create CSV for {speaker_type}: {fluent_file_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"Error processing CSV for {speaker_type} file {fluent_file_path}: {e}")
        return False

def process_file_worker(
    args: Tuple[Path, Path, Path, bool, str]
) -> Tuple[Path, bool, str]:
    """
    Worker function for parallel processing of a single file.
    
    Args:
        args: Tuple containing (input_path, output_path, output_dir, process_interviewer, model_path)
        
    Returns:
        Tuple of (input_path, success, error_message)
    """
    input_path, output_path, output_dir, process_interviewer, model_path = args
    
    try:
        # Initialize processor in worker process
        processor = DisfluencyProcessor(str(model_path))
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input file based on type
        input_type = get_file_type(input_path)
        print(f"Processing {input_type.upper()} file: {input_path.name}")
        
        if input_type == 'text':
            text = read_text_file(input_path)
        else:  # docx
            text = read_docx_file(input_path)
        
        if not text.strip():
            print(f"Warning: Skipping empty file: {input_path.name}")
            return input_path, False, "Empty file"
        
        # Process text
        clean_text = processor.process_text(text)
        
        # Clean up empty speaker sections
        clean_text = cleanup_empty_speakers(clean_text)
        
        # Write output based on output file extension
        output_type = get_file_type(output_path)
        if output_type == 'text':
            write_text_file(output_path, clean_text)
        else:  # docx
            write_docx_file(output_path, clean_text)
        
        # Separate speakers and save to respective directories
        print(f"Separating speakers for: {output_path.name}")
        separate_speakers(output_path, output_dir)
        
        # Process CSV for each speaker type
        fluent_interviews_dir = output_dir / 'fluent_interviews'
        
        # Process interviewee files
        interviewee_dir = fluent_interviews_dir / 'interviewee_fluent_interviews'
        if interviewee_dir.exists():
            for interviewee_file in interviewee_dir.glob(f"{output_path.stem}_interviewee.*"):
                if interviewee_file.is_file():
                    print(f"Processing CSV for interviewee file: {interviewee_file.name}")
                    process_csv_for_speaker(interviewee_file, output_dir, 'interviewee', process_interviewer)
        
        # Process interviewer files (if requested)
        if process_interviewer:
            interviewer_dir = fluent_interviews_dir / 'interviewer_fluent_interviews'
            if interviewer_dir.exists():
                for interviewer_file in interviewer_dir.glob(f"{output_path.stem}_interviewer.*"):
                    if interviewer_file.is_file():
                        print(f"Processing CSV for interviewer file: {interviewer_file.name}")
                        process_csv_for_speaker(interviewer_file, output_dir, 'interviewer', process_interviewer)
        
        print(f"Successfully processed: {input_path.name}")
        return input_path, True, ""
        
    except Exception as e:
        error_msg = f"Error processing {input_path.name}: {e}"
        print(error_msg)
        return input_path, False, error_msg

def process_file(
    processor: DisfluencyProcessor,
    input_path: Path,
    output_path: Path,
    output_dir: Path,
    process_interviewer: bool = False
) -> bool:
    """Process a single file with the disfluency processor and CSV conversion (sequential version)."""
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
        
        # Clean up empty speaker sections
        clean_text = cleanup_empty_speakers(clean_text)
        
        # Write output based on output file extension
        output_type = get_file_type(output_path)
        if output_type == 'text':
            write_text_file(output_path, clean_text)
        else:  # docx
            write_docx_file(output_path, clean_text)
        
        # Separate speakers and save to respective directories
        logger.info(f"Separating speakers for: {output_path.name}")
        separate_speakers(output_path, output_dir)
        
        # Process CSV for each speaker type
        fluent_interviews_dir = output_dir / 'fluent_interviews'
        
        # Process interviewee files
        interviewee_dir = fluent_interviews_dir / 'interviewee_fluent_interviews'
        if interviewee_dir.exists():
            for interviewee_file in interviewee_dir.glob(f"{output_path.stem}_interviewee.*"):
                if interviewee_file.is_file():
                    logger.info(f"Processing CSV for interviewee file: {interviewee_file.name}")
                    process_csv_for_speaker(interviewee_file, output_dir, 'interviewee', process_interviewer)
        
        # Process interviewer files (if requested)
        if process_interviewer:
            interviewer_dir = fluent_interviews_dir / 'interviewer_fluent_interviews'
            if interviewer_dir.exists():
                for interviewer_file in interviewer_dir.glob(f"{output_path.stem}_interviewer.*"):
                    if interviewer_file.is_file():
                        logger.info(f"Processing CSV for interviewer file: {interviewer_file.name}")
                        process_csv_for_speaker(interviewer_file, output_dir, 'interviewer', process_interviewer)
        
        logger.info(f"Successfully processed: {input_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return False

def main():
    # Set up argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description='Process multiple interview files with disfluency removal and CSV conversion.')
    parser.add_argument('input_dir', nargs='?', default='data/raw',
                       help='Directory containing input files (default: data/raw)')
    parser.add_argument('output_dir', nargs='?', default='data/processed',
                       help='Directory to save processed files (default: data/processed)')
    parser.add_argument('--ext', nargs='+', default=get_supported_extensions(),
                       help=f'File extensions to process (default: {get_supported_extensions()})')
    parser.add_argument('--model', default=None,
                       help='Path to the model file (default: joint-disfluency-detector-and-parser/best_models/swbd_fisher_bert_Edev.0.9078.pt)')
    parser.add_argument('--process-interviewer-to-csv', action='store_true',
                       help='Process interviewer files to CSV (default: only interviewee files are processed)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing (default: sequential processing)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes for parallel processing (default: 50% of CPU cores)')
    
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
    if args.process_interviewer_to_csv:
        logger.info("Interviewer files will be processed to CSV")
    else:
        logger.info("Only interviewee files will be processed to CSV")
    
    # Determine processing mode
    if args.parallel:
        logger.info("Using parallel processing")
        if args.workers:
            logger.info(f"Using {args.workers} worker processes")
        else:
            default_workers = max(1, mp.cpu_count() // 2)  # 50% of CPU cores, minimum 1
            logger.info(f"Using {default_workers} worker processes (50% of CPU cores)")
    else:
        logger.info("Using sequential processing")
    
    # Prepare file processing tasks
    processing_tasks = []
    for input_file in input_files:
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
        
        processing_tasks.append((input_file, output_file))
    
    # Process files
    success_count = 0
    failed_files = []
    
    if args.parallel:
        # Parallel processing
        try:
            # Determine number of workers
            num_workers = args.workers if args.workers else max(1, mp.cpu_count() // 2)
            
            # Prepare arguments for worker processes
            worker_args = [
                (input_file, output_file, output_dir, args.process_interviewer_to_csv, str(model_path))
                for input_file, output_file in processing_tasks
            ]
            
            # Process files in parallel
            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(process_file_worker, worker_args)
            
            # Collect results
            for input_path, success, error_msg in results:
                if success:
                    success_count += 1
                    logger.info(f"✅ Successfully processed: {input_path.name}")
                else:
                    failed_files.append((input_path, error_msg))
                    logger.error(f"❌ Failed to process: {input_path.name} - {error_msg}")
                    
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            logger.info("Falling back to sequential processing...")
            
            # Fallback to sequential processing
            for input_file, output_file in processing_tasks:
                logger.info(f"Processing file: {input_file.name}")
                if process_file(processor, input_file, output_file, output_dir, args.process_interviewer_to_csv):
                    success_count += 1
                else:
                    failed_files.append((input_file, "Sequential processing failed"))
    else:
        # Sequential processing
        for i, (input_file, output_file) in enumerate(processing_tasks, 1):
            logger.info(f"Processing file {i}/{len(processing_tasks)}: {input_file.name}")
            
            if process_file(processor, input_file, output_file, output_dir, args.process_interviewer_to_csv):
                success_count += 1
            else:
                failed_files.append((input_file, "Sequential processing failed"))
    
    # Print summary
    logger.info(f"\nProcessing complete!")
    logger.info(f"Successfully processed {success_count} of {len(input_files)} files")
    
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files:")
        for failed_file, error_msg in failed_files:
            logger.warning(f"  - {failed_file.name}: {error_msg}")
    
    logger.info(f"Full fluent interviews saved to: {output_dir / 'fluent_interviews' / 'full_fluent_interviews'}")
    logger.info(f"Speaker-separated content saved to:")
    logger.info(f"  - {output_dir / 'fluent_interviews' / 'interviewer_fluent_interviews'}")
    logger.info(f"  - {output_dir / 'fluent_interviews' / 'interviewee_fluent_interviews'}")
    logger.info(f"CSV files saved to:")
    logger.info(f"  - {output_dir / 'fluent_interviewer_sentence_csv'}")
    logger.info(f"  - {output_dir / 'fluent_interviewee_sentence_csv'}")

if __name__ == "__main__":
    main()
