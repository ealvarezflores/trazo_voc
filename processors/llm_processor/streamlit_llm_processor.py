#!/usr/bin/env python3
"""
Streamlit LLM Processor

This module processes validated interviewee sentences by submitting them to the Streamlit app's LLM API
and saving the responses to organized output files.
"""

import requests
import json
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamlitLLMProcessor:
    """Process sentences through Streamlit app's LLM API and save responses."""
    
    def __init__(self, streamlit_url: str = "https://gpt-voc.streamlit.app/", max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Streamlit LLM processor.
        
        Args:
            streamlit_url: The Streamlit app URL
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.streamlit_url = streamlit_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        
    def submit_to_llm(self, csv_file_path: Path, category: str) -> Optional[Dict]:
        """
        Submit a CSV file to the Streamlit app's LLM API.
        
        Args:
            csv_file_path: Path to the CSV file with sentences
            category: The product category for context
            
        Returns:
            API response as dictionary or None if failed
        """
        try:
            # Prepare the multipart form data
            with open(csv_file_path, 'rb') as f:
                files = {
                    'file': (csv_file_path.name, f, 'text/csv')
                }
                
                data = {
                    'category': category
                }
                
                # Submit to the Streamlit app
                response = self.session.post(
                    f"{self.streamlit_url}submit_llm",
                    files=files,
                    data=data,
                    timeout=120  # Longer timeout for LLM processing
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error submitting to LLM: {e}")
            return None
    
    def process_csv_file(self, input_csv_path: Path, output_dir: Path, 
                        category: str = "Software for Architects, General Contractors and Interior Designers") -> bool:
        """
        Process a CSV file through the Streamlit app's LLM API.
        
        Args:
            input_csv_path: Path to the input CSV file with sentences
            output_dir: Directory to save the processed results
            category: Product category for LLM context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if CSV file exists and has content
            if not input_csv_path.exists():
                logger.error(f"CSV file does not exist: {input_csv_path}")
                return False
            
            df = pd.read_csv(input_csv_path)
            if 'Input' not in df.columns:
                logger.error(f"CSV file does not contain 'Input' column: {input_csv_path}")
                return False
            
            sentences = df['Input'].tolist()
            logger.info(f"Found {len(sentences)} sentences to process")
            
            # Check if we need to split into chunks (Streamlit app has 1000 sentence limit)
            max_sentences_per_chunk = 1000
            if len(sentences) > max_sentences_per_chunk:
                logger.info(f"Splitting {len(sentences)} sentences into chunks of {max_sentences_per_chunk}")
                chunks = [sentences[i:i + max_sentences_per_chunk] 
                         for i in range(0, len(sentences), max_sentences_per_chunk)]
            else:
                chunks = [sentences]
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            all_results = []
            
            # Process each chunk
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} sentences")
                
                # Create temporary CSV for this chunk
                chunk_df = pd.DataFrame({'Input': chunk})
                temp_csv_path = output_dir / f"temp_chunk_{i+1}.csv"
                chunk_df.to_csv(temp_csv_path, index=False)
                
                try:
                    # Submit chunk to LLM
                    result = self.submit_to_llm(temp_csv_path, category)
                    
                    if result:
                        # Add chunk information to result
                        result['chunk_index'] = i + 1
                        result['chunk_size'] = len(chunk)
                        result['original_csv'] = input_csv_path.name
                        all_results.append(result)
                        logger.info(f"Successfully processed chunk {i+1}")
                    else:
                        logger.error(f"Failed to process chunk {i+1}")
                        
                finally:
                    # Clean up temporary file
                    if temp_csv_path.exists():
                        temp_csv_path.unlink()
                
                # Delay between chunks to avoid overwhelming the API
                if i < len(chunks) - 1:
                    time.sleep(2)
            
            # Save all results
            if all_results:
                output_file = output_dir / f"{input_csv_path.stem}_llm_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Processing complete. Results saved to: {output_file}")
                return True
            else:
                logger.error("No results obtained from LLM processing")
                return False
                
        except Exception as e:
            logger.error(f"Error processing CSV file {input_csv_path}: {e}")
            return False
    
    def process_directory(self, input_dir: Path, output_dir: Path,
                         category: str = "Software for Architects, General Contractors and Interior Designers") -> bool:
        """
        Process all CSV files in a directory.
        
        Args:
            input_dir: Directory containing CSV files to process
            output_dir: Directory to save processed results
            category: Product category for LLM context
            
        Returns:
            True if all files processed successfully, False otherwise
        """
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return False
        
        csv_files = list(input_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in: {input_dir}")
            return False
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        success_count = 0
        for csv_file in csv_files:
            logger.info(f"Processing file: {csv_file.name}")
            
            # Create subdirectory for this file's results
            file_output_dir = output_dir / csv_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.process_csv_file(csv_file, file_output_dir, category):
                success_count += 1
            else:
                logger.error(f"Failed to process: {csv_file.name}")
        
        logger.info(f"Directory processing complete: {success_count}/{len(csv_files)} files successful")
        return success_count == len(csv_files)


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sentences through Streamlit app LLM API')
    parser.add_argument('input_path', help='Input CSV file or directory')
    parser.add_argument('output_dir', help='Output directory for processed results')
    parser.add_argument('--streamlit-url', default="https://gpt-voc.streamlit.app/",
                       help='Streamlit app URL')
    parser.add_argument('--category', default="Software for Architects, General Contractors and Interior Designers",
                       help='Product category for LLM context')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retry attempts')
    parser.add_argument('--retry-delay', type=float, default=1.0, help='Delay between retries in seconds')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = StreamlitLLMProcessor(
        streamlit_url=args.streamlit_url,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    
    # Process based on input type
    if input_path.is_file():
        success = processor.process_csv_file(input_path, output_dir, args.category)
    elif input_path.is_dir():
        success = processor.process_directory(input_path, output_dir, args.category)
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    if success:
        logger.info("Processing completed successfully")
    else:
        logger.error("Processing failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
