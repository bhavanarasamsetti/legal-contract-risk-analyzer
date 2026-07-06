"""One-time script: parse PDFs and build the local BM25 index.

This script runs the frozen parser pipeline and persists the resulting
chunks to ``data/bm25_index.json``.  Run it after ``scripts/ingest.py``
whenever the contract corpus changes.

Usage::

    python scripts/build_bm25_index.py

Custom contracts directory::

    python scripts/build_bm25_index.py --contracts-dir /path/to/contracts

Custom output path::

    python scripts/build_bm25_index.py --output data/custom_bm25.json
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.bm25_index import BM25Index, DEFAULT_INDEX_PATH
from app.chunker import LegalSemanticChunker
from app.document_assembler import DocumentAssembler
from app.pdf_loader import PDFLoader
from app.text_preprocessor import TextPreprocessor

DEFAULT_CONTRACTS_DIR = PROJECT_ROOT / "contracts" / "sample_contracts"
_SEP = "─" * 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the local BM25 index from contract PDFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=DEFAULT_CONTRACTS_DIR,
        metavar="PATH",
        help="Directory containing the PDF files to index.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        metavar="PATH",
        help="Destination path for the persisted BM25 JSON index.",
    )
    return parser.parse_args()


def _run_parser(contracts_dir: Path) -> list:
    """Load PDFs and return all chunks from the frozen parser pipeline."""
    pdf_files = sorted(contracts_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"  ✗  No PDF files found in: {contracts_dir}")
        sys.exit(1)

    loader = PDFLoader()
    preprocessor = TextPreprocessor()
    assembler = DocumentAssembler()
    chunker = LegalSemanticChunker()

    all_pages = []
    for pdf_path in pdf_files:
        try:
            raw_pages = loader.load_pdf(str(pdf_path))
            clean_pages = preprocessor.preprocess_pages(raw_pages)
            all_pages.extend(clean_pages)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"  ⚠   Skipping '{pdf_path.name}': {exc}")

    if not all_pages:
        print("  ✗  No pages were successfully loaded.  Aborting.")
        sys.exit(1)

    documents = assembler.assemble_documents(all_pages)
    return chunker.chunk_documents(documents)


def main() -> None:
    args = _parse_args()

    print(f"\n[1/2] Parsing documents  {_SEP}")
    print(f"  Source : {args.contracts_dir}")

    chunks = _run_parser(args.contracts_dir)
    print(f"  Total  : {len(chunks)} chunks")

    print(f"\n[2/2] Building BM25 index  {_SEP}")
    try:
        index = BM25Index.from_chunks(chunks)
        index.save(args.output)
    except ValueError as exc:
        print(f"  ✗  {exc}")
        sys.exit(1)

    print(f"  ✓  Saved {index.size} chunks to {args.output}")
    print(f"\n{'─' * 56}")
    print("  BM25 index build complete.")
    print(f"{'─' * 56}\n")


if __name__ == "__main__":
    main()
