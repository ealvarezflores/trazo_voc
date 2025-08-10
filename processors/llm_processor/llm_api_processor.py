#!/usr/bin/env python3
"""
LLM API Processor

This module processes validated interviewee sentences by submitting them to an LLM API
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMAPIProcessor:
    """Process sentences through LLM API and save responses."""
    
    def __init__(self, api_url: str, api_key: str, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the LLM API processor.
        
        Args:
            api_url: The LLM API endpoint URL
            api_key: API key for authentication
            max_retries: Maximum number of retry attempts for failed requests
            retry_delay: Delay between retry attempts in seconds
        """
        self.api_url = api_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
    def call_llm_api(self, sentence: str, category: str) -> Optional[Dict]:
        """
        Call the LLM API with a single sentence.
        
        Args:
            sentence: The sentence to process
            category: The product category for context
            
        Returns:
            API response as dictionary or None if failed
        """
        payload = {
            "sentence": sentence,
            "category": category
        }
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"API call failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                
        logger.error(f"Failed to call API after {self.max_retries} attempts for sentence: {sentence[:50]}...")
        return None
    
    def process_sentence(self, sentence: str, category: str) -> Dict:
        """
        Process a single sentence through the LLM API.
        
        Args:
            sentence: The sentence to process
            category: The product category for context
            
        Returns:
            Dictionary containing original sentence and LLM response
        """
        result = {
            'original_sentence': sentence,
            'category': category,
            'llm_response': None,
            'success': False,
            'error': None
        }
        
        try:
            api_response = self.call_llm_api(sentence, category)
            
            if api_response:
                result['llm_response'] = api_response
                result['success'] = True
            else:
                result['error'] = 'API call failed'
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing sentence: {e}")
            
        return result
    
    def process_csv_file(self, input_csv_path: Path, output_dir: Path, 
                        category: str = "Software for Architects, General Contractors and Interior Designers") -> bool:
        """
        Process all sentences in a CSV file through the LLM API.
        
        Args:
            input_csv_path: Path to the input CSV file with sentences
            output_dir: Directory to save the processed results
            category: Product category for LLM context
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read the CSV file
            logger.info(f"Reading CSV file: {input_csv_path}")
            df = pd.read_csv(input_csv_path)
            
            if 'Input' not in df.columns:
                logger.error(f"CSV file does not contain 'Input' column: {input_csv_path}")
                return False
            
            sentences = df['Input'].tolist()
            logger.info(f"Found {len(sentences)} sentences to process")
            
            # Process each sentence
            results = []
            for sentence in tqdm(sentences, desc="Processing sentences"):
                if pd.isna(sentence) or not str(sentence).strip():
                    continue
                    
                result = self.process_sentence(str(sentence).strip(), category)
                results.append(result)
                
                # Small delay to avoid overwhelming the API
                time.sleep(0.1)
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            output_file = output_dir / f"{input_csv_path.stem}_llm_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Create summary CSV
            summary_data = []
            for result in results:
                if result['success'] and result['llm_response']:
                    summary_data.append({
                        'original_sentence': result['original_sentence'],
                        'llm_response': json.dumps(result['llm_response']),
                        'category': result['category']
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_csv = output_dir / f"{input_csv_path.stem}_llm_summary.csv"
                summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
                logger.info(f"Saved summary to: {summary_csv}")
            
            # Log statistics
            successful = sum(1 for r in results if r['success'])
            failed = len(results) - successful
            logger.info(f"Processing complete: {successful} successful, {failed} failed")
            logger.info(f"Results saved to: {output_file}")
            
            return True
            
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
    
    parser = argparse.ArgumentParser(description='Process sentences through LLM API')
    parser.add_argument('input_path', help='Input CSV file or directory')
    parser.add_argument('output_dir', help='Output directory for processed results')
    parser.add_argument('--api-url', required=True, help='LLM API URL')
    parser.add_argument('--api-key', required=True, help='LLM API key')
    parser.add_argument('--category', default="Software for Architects, General Contractors and Interior Designers",
                       help='Product category for LLM context')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum retry attempts')
    parser.add_argument('--retry-delay', type=float, default=1.0, help='Delay between retries in seconds')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = LLMAPIProcessor(
        api_url=args.api_url,
        api_key=args.api_key,
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
