import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from nltk import Tree

# This module wraps the upstream repo `pariajm/joint-disfluency-detector-and-parser`.
# It expects the repo to be cloned locally and a pretrained `.pt` model downloaded.
# We then call its CLI to produce constituency parse trees with EDITED spans and
# remove words inside EDITED spans for minimal light-editing.


def _run_cmd(cmd: str, cwd: Optional[Path] = None) -> None:
    proc = subprocess.run(
        shlex.split(cmd), cwd=str(cwd) if cwd else None, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if proc.stdout:
        print(proc.stdout)


def run_joint_parser(
    sentences: List[str],
    repo_path: Path,
    model_pt_path: Path,
    output_dir: Optional[Path] = None,
) -> List[str]:
    """
    Call upstream parser to get 1 constituency tree per sentence.

    Returns list of bracketed trees as strings (one per sentence).
    """
    repo_path = repo_path.resolve()
    if not repo_path.exists():
        raise FileNotFoundError(f"Repo path not found: {repo_path}")
    if not model_pt_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_pt_path}")

    work_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="joint_parse_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    input_path = work_dir / "raw_sentences.txt"
    output_path = work_dir / "parsed_sentences.txt"

    # Upstream expects one sentence per line; better performance if punctuation stripped and clitics split.
    with open(input_path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s.strip() + "\n")

    cmd = (
        f"python3 {repo_path}/src/main.py parse "
        f"--input-path {input_path} "
        f"--output-path {output_path} "
        f"--model-path-base {model_pt_path}"
    )
    _run_cmd(cmd, cwd=repo_path)

    if not output_path.exists():
        raise RuntimeError("Parser did not produce output file")

    trees: List[str] = []
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trees.append(line)
    return trees


def _remove_edited_from_tree_str(tree_str: str) -> List[str]:
    """Parse a Penn-style tree, drop leaves that are under EDITED nodes, return remaining tokens.

    Uses leaf indices (positions) rather than token string matching to avoid removing
    duplicate tokens that are not within EDITED spans.
    """
    try:
        t = Tree.fromstring(tree_str)
    except Exception:
        # If parsing fails, return a naive tokenization by whitespace as fallback
        return tree_str.replace("(", " ").replace(")", " ").split()

    # Map absolute leaf positions to indices in t.leaves()
    leaves = t.leaves()
    leaf_positions = list(t.treepositions('leaves'))  # tuples indicating path to each leaf
    pos_to_index = {pos: idx for idx, pos in enumerate(leaf_positions)}

    remove_indices = set()
    for pos in leaf_positions:
        # Check ancestors for EDITED label
        # Ancestors are prefixes of the position path
        edited = False
        for i in range(1, len(pos)):
            ancestor = pos[:i]
            try:
                node = t[ancestor]
                if isinstance(node, Tree) and node.label() == 'EDITED':
                    edited = True
                    break
            except Exception:
                continue
        if edited:
            remove_indices.add(pos_to_index[pos])

    kept_tokens: List[str] = [tok for i, tok in enumerate(leaves) if i not in remove_indices]
    return kept_tokens


def clean_sentences_from_trees(tree_strs: List[str]) -> List[str]:
    """Remove EDITED spans and lightly normalize spacing around punctuation."""
    cleaned: List[str] = []
    for ts in tree_strs:
        tokens = _remove_edited_from_tree_str(ts)
        if not tokens:
            cleaned.append("")
            continue
        # Join with simple punctuation rules (no paraphrasing)
        text = " ".join(tokens)
        # Fix common spaces before punctuation
        for p in [",", ".", "!", "?", ":", ";"]:
            text = text.replace(f" {p}", p)
        # Capitalize sentence start minimally
        if text:
            text = text[0].upper() + text[1:]
        cleaned.append(text)
    return cleaned


def joint_light_edit(
    sentences: List[str],
    repo_path: Path,
    model_pt_path: Path,
    output_dir: Optional[Path] = None,
) -> List[str]:
    trees = run_joint_parser(sentences, repo_path, model_pt_path, output_dir)
    return clean_sentences_from_trees(trees)
