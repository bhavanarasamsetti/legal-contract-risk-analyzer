"""End-to-end smoke test for the document ingestion and chunking pipeline.

Run from the project root with:

    python tests/test_chunker.py

The script exercises the full pipeline:

    PDFLoader → TextPreprocessor → DocumentAssembler → LegalSemanticChunker

and prints a human-readable summary of every generated chunk.
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `app.*` imports resolve when
# the script is executed directly (i.e. without `python -m`).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import ChunkData, LegalSemanticChunker
from app.document_assembler import DocumentAssembler
from app.pdf_loader import PDFLoader
from app.text_preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTRACTS_DIR = PROJECT_ROOT / "contracts" / "sample_contracts"
PREVIEW_LENGTH = 200
CHUNK_SEPARATOR = "-" * 40
DOC_SEPARATOR = "=" * 40

# List specific PDF filenames to test, or leave empty to test all PDFs.
#
# Examples:
#   TEST_PDFS = ["atlassian_customer_dpa.pdf"]
#   TEST_PDFS = ["employment_agreement.pdf"]
#   TEST_PDFS = ["data_processing_agreement.pdf"]
#   TEST_PDFS = []  ← runs all PDFs in CONTRACTS_DIR
TEST_PDFS: list[str] = [
    "atlassian_customer_dpa.pdf",
    "data_processing_agreement.pdf",
    "employment_agreement.pdf",
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def print_chunk(chunk: ChunkData) -> None:
    """Print a formatted summary block for a single chunk.

    Args:
        chunk: A fully populated :class:`~app.chunker.ChunkData` dict.
    """
    preview = chunk["chunk_text"][:PREVIEW_LENGTH].replace("\n", " ")

    print(CHUNK_SEPARATOR)
    print(f"Chunk ID:       {chunk['chunk_id']}")
    print(f"Document:       {chunk['document_name']}")
    print(f"Pages:          {chunk['pages']}")
    print(f"Section:        {chunk['section']}")
    print(f"Section Title:  {chunk['section_title'] or '(none)'}")
    print(f"Parent Section: {chunk['parent_section'] or '(none)'}")
    print(f"Characters:     {len(chunk['chunk_text'])}")
    print(f"Preview:        {preview}")
    print(CHUNK_SEPARATOR)


def print_document_header(document_name: str, chunk_count: int) -> None:
    """Print a document-level header before its chunks are listed.

    Args:
        document_name: Filename of the PDF being summarised.
        chunk_count: Number of chunks generated from this document.
    """
    print(f"\n{DOC_SEPARATOR}")
    print(f"Document : {document_name}")
    print(f"Chunks   : {chunk_count}")
    print(DOC_SEPARATOR)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """Execute the full ingestion pipeline and print a chunk-level summary.

    Steps:
        1. Discover all PDF files in ``CONTRACTS_DIR``.
        2. Load each PDF with :class:`~app.pdf_loader.PDFLoader`.
        3. Clean pages with :class:`~app.text_preprocessor.TextPreprocessor`.
        4. Assemble per-document structures with
           :class:`~app.document_assembler.DocumentAssembler`.
        5. Generate semantic chunks with
           :class:`~app.chunker.LegalSemanticChunker`.
        6. Print a readable summary for every chunk.
    """
    if TEST_PDFS:
        pdf_files = [CONTRACTS_DIR / name for name in TEST_PDFS]
        missing = [p for p in pdf_files if not p.is_file()]
        if missing:
            for p in missing:
                print(f"[WARNING] File not found: {p}")
        pdf_files = [p for p in pdf_files if p.is_file()]
    else:
        pdf_files = sorted(CONTRACTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARNING] No PDF files found in: {CONTRACTS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{CONTRACTS_DIR}'")

    # Instantiate pipeline components once — all are stateless
    loader = PDFLoader()
    preprocessor = TextPreprocessor()
    assembler = DocumentAssembler()
    chunker = LegalSemanticChunker()

    # --- Stage 1 & 2: Load and clean all pages across every PDF ---
    all_pages = []

    for pdf_path in pdf_files:
        try:
            raw_pages = loader.load_pdf(str(pdf_path))
            clean_pages = preprocessor.preprocess_pages(raw_pages)
            all_pages.extend(clean_pages)
            print(f"  Loaded : {pdf_path.name}  ({len(clean_pages)} pages after cleaning)")

        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"  [ERROR] Could not load '{pdf_path.name}': {exc}")

    if not all_pages:
        print("[ERROR] No pages were successfully loaded. Aborting.")
        return

    # --- Stage 3: Assemble pages into per-document structures ---
    documents = assembler.assemble_documents(all_pages)
    print(f"\nAssembled {len(documents)} document(s).\n")

    # --- Stage 4: Chunk each document semantically ---
    all_chunks = chunker.chunk_documents(documents)

    # --- Stage 5: Print chunk summaries grouped by document ---
    chunks_by_doc: dict[str, list[ChunkData]] = {}
    for chunk in all_chunks:
        chunks_by_doc.setdefault(chunk["document_name"], []).append(chunk)

    for document_name, doc_chunks in chunks_by_doc.items():
        print_document_header(document_name, len(doc_chunks))
        for chunk in doc_chunks:
            print_chunk(chunk)

    # --- Final totals ---
    print(f"\n{'=' * 40}")
    print(f"Total Documents : {len(documents)}")
    print(f"Total Chunks    : {len(all_chunks)}")
    print(f"{'=' * 40}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n[FATAL] Unexpected error: {exc}")
        sys.exit(1)
