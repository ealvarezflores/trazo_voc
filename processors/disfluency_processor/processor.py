import os
import sys
import torch
from pathlib import Path
from typing import List, Optional

# Add the joint-disfluency-detector-and-parser/src directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "joint-disfluency-detector-and-parser"))
sys.path.insert(0, str(project_root / "joint-disfluency-detector-and-parser" / "src"))

# Now import the required modules
from parse_nk import NKChartParser
import parse_nk

class DisfluencyProcessor:
    def __init__(self, model_path: str):
        """Initialize the disfluency processor with the specified model.
        
        Args:
            model_path: Path to the trained model file (.pt)
        """
        # Set device (MPS for Apple Silicon, CPU as fallback)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using MPS (Metal Performance Shaders) for acceleration")
        else:
            self.device = torch.device('cpu')
            print("Using CPU (no GPU acceleration available)")
            
        # Load the model
        self.parser = self._load_model(model_path)
        
        # Set model to evaluation mode
        self.parser.eval()
        
        # Get dummy tag for parsing
        if hasattr(self.parser, 'tag_vocab'):
            if 'UNK' in self.parser.tag_vocab.indices:
                self.dummy_tag = 'UNK'
            else:
                self.dummy_tag = self.parser.tag_vocab.value(0)
        else:
            self.dummy_tag = 'UNK'
    
    def _load_model(self, model_path: str) -> NKChartParser:
        """Load the pre-trained disfluency detection model.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded NKChartParser model
        """
        print(f"Loading model from {model_path}...")
        
        # Load model info (on CPU first) with weights_only=False for compatibility
        # Note: Setting weights_only=False is required for this model but be cautious with untrusted files
        info = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Initialize model using from_spec (handles device placement internally)
        parser = NKChartParser.from_spec(info['spec'], info['state_dict'])
        
        # Move model to the appropriate device
        parser = parser.to(self.device)
        print(f"Model loaded on device: {next(parser.parameters()).device}")
        
        return parser
    
    def process_text(self, text):
        """
        Process a single text string to detect disfluencies.
        
        Args:
            text (str): Input text to process
            
        Returns:
            list: List of processed sentences with disfluency annotations
        """
        # Split into sentences (simple split on periods for now)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Tokenize sentences (simple whitespace tokenizer for now)
        tokenized_sentences = [sentence.split() for sentence in sentences]
        
        # Prepare input with dummy tags
        tagged_sentences = [[(self.dummy_tag, word) for word in sentence] 
                          for sentence in tokenized_sentences]
        
        # Process in batches (single batch for now)
        with torch.no_grad():
            parsed_trees, _ = self.parser.parse_batch(tagged_sentences)
        
        # Convert trees to linearized format
        results = []
        for tree in parsed_trees:
            results.append(tree.convert().linearize())
            
        return results

def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    model_path = base_dir / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    input_path = base_dir / 'data' / 'raw' / 'input.txt'
    output_path = base_dir / 'data' / 'processed' / 'output.txt'
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    processor = DisfluencyProcessor(str(model_path))
    
    # Read input text
    with open(input_path, 'r') as f:
        text = f.read().strip()
    
    # Process text
    print("Processing text...")
    results = processor.process_text(text)
    
    # Write results
    with open(output_path, 'w') as f:
        for result in results:
            f.write(f"{result}\n")
    
    print(f"Processing complete. Results written to {output_path}")

if __name__ == "__main__":
    main()
