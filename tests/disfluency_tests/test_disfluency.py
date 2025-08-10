#!/usr/bin/env python3

import sys
import torch
from pathlib import Path

# Add the joint-disfluency-detector-and-parser/src directory to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "joint-disfluency-detector-and-parser"))
sys.path.insert(0, str(project_root / "joint-disfluency-detector-and-parser" / "src"))

from parse_nk import NKChartParser
from trees import InternalTreebankNode, LeafTreebankNode

def test_disfluency_detection():
    """Test the disfluency detection with a simple sentence containing obvious disfluencies."""
    
    # Force using CPU
    device = torch.device("cpu")
    print("Using CPU for testing")
    
    # Load the model
    model_path = project_root / 'joint-disfluency-detector-and-parser' / 'best_models' / 'swbd_fisher_bert_Edev.0.9078.pt'
    print(f"Loading model from {model_path}...")
    
    info = torch.load(model_path, map_location='cpu', weights_only=False)
    parser = NKChartParser.from_spec(info['spec'], info['state_dict'])
    parser = parser.to(device)
    parser.eval()
    
    # Get dummy tag
    if hasattr(parser, 'tag_vocab'):
        if 'UNK' in parser.tag_vocab.indices:
            dummy_tag = 'UNK'
        else:
            dummy_tag = parser.tag_vocab.value(0)
    else:
        dummy_tag = 'UNK'
    
    # Test sentence with obvious disfluencies
    test_sentence = "There is something called shop drawings which is like a technical drawing thing yeah right that you need to have some technical knowledge of course yes"
    
    print(f"\nTesting with sentence: '{test_sentence}'")
    
    # Tokenize
    words = test_sentence.split()
    tagged_sentence = [(dummy_tag, word) for word in words]
    
    print(f"Tagged sentence: {tagged_sentence}")
    
    # Parse
    with torch.no_grad():
        parsed_trees, scores = parser.parse_batch([tagged_sentence])
    
    if parsed_trees and parsed_trees[0]:
        tree = parsed_trees[0]
        print(f"\nParse tree: {tree}")
        
        # Convert to treebank format
        treebank_tree = tree.convert()
        print(f"\nTreebank tree: {treebank_tree}")
        
        # Print the tree structure
        print(f"\nTree structure:")
        print_tree_structure(treebank_tree, 0)
        
        # Test disfluency detection
        print(f"\nDisfluency detection test:")
        test_disfluency_detection_on_tree(treebank_tree)
        
        # Extract clean text
        clean_text = extract_clean_text(treebank_tree)
        print(f"\nClean text: '{clean_text}'")
        
    else:
        print("No parse tree generated!")

def print_tree_structure(node, depth=0):
    """Print the tree structure with indentation."""
    indent = "  " * depth
    if isinstance(node, LeafTreebankNode):
        print(f"{indent}Leaf: {node.tag} -> '{node.word}'")
    else:
        print(f"{indent}Internal: {node.label}")
        for child in node.children:
            print_tree_structure(child, depth + 1)

def test_disfluency_detection_on_tree(node):
    """Test disfluency detection on each node."""
    if isinstance(node, LeafTreebankNode):
        return
    
    label = node.label.upper()
    is_disfluent = any(disfluency in label for disfluency in ['EDITED', 'PRN', 'UH', 'INTJ'])
    
    if is_disfluent:
        print(f"  Found disfluency node: {node.label}")
        # Print the words under this node
        words = []
        for child in node.leaves():
            words.append(child.word)
        print(f"    Words: {' '.join(words)}")
    
    for child in node.children:
        test_disfluency_detection_on_tree(child)

def extract_clean_text(node):
    """Extract clean text from a tree node, skipping disfluent segments."""
    if isinstance(node, LeafTreebankNode):
        return node.word
        
    # Check if this is a disfluency node
    label = node.label.upper()
    is_disfluent = any(disfluency in label for disfluency in ['EDITED', 'PRN', 'UH', 'INTJ'])
    
    if is_disfluent:
        print(f"  Skipping disfluent node: {node.label}")
        return ""
        
    # Process children and join their text
    parts = []
    for child in node.children:
        child_text = extract_clean_text(child)
        if child_text:
            parts.append(child_text)
            
    return " ".join(parts)

if __name__ == "__main__":
    test_disfluency_detection()
