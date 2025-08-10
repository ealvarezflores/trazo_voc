#!/usr/bin/env python3

import sys
from pathlib import Path

# Add the processors directory to the path
sys.path.insert(0, str(Path(__file__).parent / "processors" / "disfluency_processor"))

from disfluency_processor import DisfluencyProcessor
import torch

def debug_parser():
    """Debug what the parser is detecting in the text."""
    
    # Initialize processor
    model_path = Path(__file__).parent / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    processor = DisfluencyProcessor(str(model_path))
    
    # Test with a simple sentence containing obvious disfluencies
    test_text = "There is something called shop drawings which is like a technical drawing thing yeah right that you need to have some technical knowledge of course yes"
    
    print("Original text:")
    print(test_text)
    print("\n" + "="*80 + "\n")
    
    # Process through the parser to see what it detects
    words = test_text.split()
    tagged_sentence = [(processor.dummy_tag, word) for word in words]
    
    with torch.no_grad():
        parsed_trees, _ = processor.parser.parse_batch([tagged_sentence])
    
    if parsed_trees and parsed_trees[0]:
        treebank_tree = parsed_trees[0].convert()
        
        print("Parse tree structure:")
        print_tree_structure(treebank_tree, max_depth=3)
        
        print("\n" + "="*80 + "\n")
        
        print("Disfluent nodes found:")
        find_disfluent_nodes(treebank_tree, processor)
        
        print("\n" + "="*80 + "\n")
        
        print("Clean text extracted:")
        clean_text = processor._extract_clean_text(treebank_tree)
        print(clean_text)

def print_tree_structure(node, depth=0, max_depth=3):
    """Print the tree structure up to a certain depth."""
    if depth > max_depth:
        return
        
    indent = "  " * depth
    if hasattr(node, 'word'):
        print(f"{indent}LEAF: {node.word}")
    else:
        print(f"{indent}NODE: {node.label}")
        for child in node.children:
            print_tree_structure(child, depth + 1, max_depth)

def find_disfluent_nodes(node, processor):
    """Find and print all disfluent nodes in the tree."""
    if hasattr(node, 'word'):
        return
        
    if processor._is_disfluent_node(node):
        print(f"DISFLUENT: {node.label}")
        # Print the words in this disfluent node
        words = []
        collect_words(node, words)
        print(f"  Words: {' '.join(words)}")
    
    for child in node.children:
        find_disfluent_nodes(child, processor)

def collect_words(node, words):
    """Collect all words from a node and its children."""
    if hasattr(node, 'word'):
        words.append(node.word)
    else:
        for child in node.children:
            collect_words(child, words)

if __name__ == "__main__":
    debug_parser()
