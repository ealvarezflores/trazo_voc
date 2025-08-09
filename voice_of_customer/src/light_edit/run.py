import argparse
from pathlib import Path
import sys
import spacy

from ..preprocess import TranscriptPreprocessor
from .joint_parser import joint_light_edit


def process_docx_file(docx_path: Path, repo_path: Path, model_path: Path, out_dir: Path) -> Path:
    pre = TranscriptPreprocessor()
    text = pre.read_docx(str(docx_path))

    # Use spaCy sentence segmentation (already loaded inside TranscriptPreprocessor)
    doc = pre.nlp(text)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    cleaned_sentences = joint_light_edit(sentences, repo_path, model_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{docx_path.stem}_cleaned.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_sentences))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Light-edit transcripts via Joint Disfluency Detector/Parser (EDITED removal)")
    parser.add_argument("--in", dest="in_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "raw"), help="Input directory containing .docx files")
    parser.add_argument("--out", dest="out_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data" / "processed"), help="Output directory for cleaned .txt files")
    parser.add_argument("--repo", dest="repo_path", type=str, required=True, help="Path to cloned pariajm/joint-disfluency-detector-and-parser repo")
    parser.add_argument("--model", dest="model_path", type=str, required=True, help="Path to pretrained .pt model (e.g., swbd_fisher_bert_Edev.0.9078.pt)")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    repo_path = Path(args.repo_path)
    model_path = Path(args.model_path)

    if not in_dir.exists():
        print(f"Input directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    docx_files = sorted(in_dir.glob("*.docx"))
    if not docx_files:
        print(f"No .docx files found in {in_dir}")
        sys.exit(0)

    print(f"Found {len(docx_files)} .docx files. Running joint parser light-edit...")
    for p in docx_files:
        try:
            out_path = process_docx_file(p, repo_path, model_path, out_dir)
            print(f"✔ Cleaned: {p.name} -> {out_path.name}")
        except Exception as e:
            print(f"✖ Error processing {p.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
