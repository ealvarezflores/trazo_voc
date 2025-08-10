#!/usr/bin/env python3
"""
Batch Text to CSV Processor

This module provides batch processing functionality for converting multiple
text/docx files to CSV format using the same logic as the Streamlit app.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict
from text_to_csv_processor import process_file, read_file_content, process_text_document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Batch processor for converting multiple text/docx files to CSV format.
    
    Features:
    - Processes multiple files in a directory
    - Creates individual CSV files for each input file
    - Creates consolidated CSV with all sentences
    - Supports both TXT and DOCX input formats
    - Maintains file source tracking
    """
    
    def __init__(self, sentence_min_len: int = 4, sentence_max_len: int = 200):
        """
        Initialize the batch processor.
        
        Args:
            sentence_min_len: Minimum words per sentence
            sentence_max_len: Maximum words per sentence
        """
        self.sentence_min_len = sentence_min_len
        self.sentence_max_len = sentence_max_len
        self.all_sentences = []
        self.processed_files = 0
        self.total_sentences = 0
        self.total_discarded = 0
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return ['.txt', '.docx']
    
    def find_input_files(self, input_dir: Path) -> List[Path]:
        """
        Find all supported input files in a directory.
        
        Args:
            input_dir: Input directory path
            
        Returns:
            List of input file paths
        """
        input_files = []
        supported_extensions = self.get_supported_extensions()
        
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                input_files.append(file_path)
        
        return sorted(input_files)
    
    def process_single_file(self, input_path: Path, output_dir: Path) -> bool:
        """
        Process a single file and create its CSV output.
        
        Args:
            input_path: Input file path
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create output filename
            output_filename = f"{input_path.stem}_sentences.csv"
            output_path = output_dir / output_filename
            
            # Create discarded sentences output path
            discarded_output_path = output_dir / f"{input_path.stem}_sentences_discarded.csv"
            
            # Process the file
            success = process_file(
                input_path, 
                output_path, 
                self.sentence_min_len, 
                self.sentence_max_len, 
                discarded_output_path
            )
            
            if success:
                # Read the processed content to collect sentences for consolidation
                text = read_file_content(input_path)
                filtered_sentences, discarded_sentences = process_text_document(
                    text, self.sentence_min_len, self.sentence_max_len
                )
                
                # Add to consolidated list with file source tracking
                for sentence in filtered_sentences:
                    self.all_sentences.append({
                        'Input': sentence,
                        'File_Source': input_path.name,
                        'Word_Count': len(sentence.split())
                    })
                
                self.processed_files += 1
                self.total_sentences += len(filtered_sentences)
                self.total_discarded += len(discarded_sentences)
                
                logger.info(f"Successfully processed {input_path.name} -> {len(filtered_sentences)} sentences")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing {input_path}: {e}")
            return False
    
    def create_consolidated_csv(self, output_dir: Path, filename: str = "all_sentences.csv") -> bool:
        """
        Create a consolidated CSV with all sentences from all files.
        
        Args:
            output_dir: Output directory path
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = output_dir / filename
            
            # Write consolidated CSV
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Input', 'File_Source', 'Word_Count']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for sentence_data in self.all_sentences:
                    writer.writerow(sentence_data)
            
            logger.info(f"Created consolidated CSV: {output_path}")
            logger.info(f"Total files processed: {self.processed_files}")
            logger.info(f"Total sentences: {self.total_sentences}")
            logger.info(f"Total discarded: {self.total_discarded}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating consolidated CSV: {e}")
            return False
    
    def process_directory(self, input_dir: Path, output_dir: Path, create_consolidated: bool = True) -> Dict[str, int]:
        """
        Process all files in a directory.
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            create_consolidated: Whether to create consolidated CSV
            
        Returns:
            Dictionary with processing statistics
        """
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        input_files = self.find_input_files(input_dir)
        
        if not input_files:
            logger.warning(f"No supported files found in {input_dir}")
            return {"processed": 0, "failed": 0, "total_sentences": 0, "total_discarded": 0}
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process each file
        processed_count = 0
        failed_count = 0
        
        for i, input_file in enumerate(input_files, 1):
            logger.info(f"Processing file {i}/{len(input_files)}: {input_file.name}")
            
            success = self.process_single_file(input_file, output_dir)
            
            if success:
                processed_count += 1
            else:
                failed_count += 1
        
        # Create consolidated CSV if requested
        if create_consolidated and self.all_sentences:
            self.create_consolidated_csv(output_dir)
        
        # Return statistics
        stats = {
            "processed": processed_count,
            "failed": failed_count,
            "total_sentences": self.total_sentences,
            "total_discarded": self.total_discarded,
            "total_files": len(input_files)
        }
        
        logger.info(f"Processing complete: {processed_count} successful, {failed_count} failed")
        return stats


def main():
    """Main function for command-line usage."""
    if len(sys.argv) < 3:
        print("Usage: python batch_processor.py <input_directory> <output_directory> [min_words] [max_words]")
        print("Example: python batch_processor.py data/raw data/processed/csv_sentences 4 200")
        sys.exit(1)
    
    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    
    # Optional parameters
    sentence_min_len = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    sentence_max_len = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)
    
    # Initialize batch processor
    processor = BatchProcessor(
        sentence_min_len=sentence_min_len,
        sentence_max_len=sentence_max_len
    )
    
    # Process directory
    stats = processor.process_directory(input_dir, output_dir)
    
    # Print results
    print(f"\nProcessing complete!")
    print(f"Files processed: {stats['processed']}/{stats['total_files']}")
    print(f"Total sentences: {stats['total_sentences']}")
    print(f"Total discarded: {stats['total_discarded']}")
    print(f"Output directory: {output_dir}")
    
    if stats['failed'] > 0:
        print(f"Warning: {stats['failed']} files failed to process")
        sys.exit(1)


if __name__ == "__main__":
    main()
