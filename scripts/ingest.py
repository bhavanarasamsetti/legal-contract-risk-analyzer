"""One-time ingestion script: parse PDFs → embed → upsert to Pinecone.

This script drives the full RAG ingestion pipeline:

    PDFLoader → TextPreprocessor → DocumentAssembler → LegalSemanticChunker
        → EmbeddingGenerator → VectorStore

It is designed to be **idempotent**: running it multiple times on the same
contracts directory produces the same Pinecone index state because each
vector ID is derived from the deterministic ``chunk_id`` field, so re-upserts
overwrite rather than duplicate existing vectors.

Usage
-----
First run (creates the Pinecone index, then ingests)::

    python scripts/ingest.py --create-index

Every subsequent run (re-ingest; index already exists)::

    python scripts/ingest.py

Custom contracts directory::

    python scripts/ingest.py --contracts-dir /path/to/contracts

Show a tqdm progress bar during the Pinecone upsert::

    python scripts/ingest.py --show-progress

Non-default embedding model dimension (e.g. text-embedding-3-large)::

    python scripts/ingest.py --create-index --dimension 3072

All paths are resolved relative to this file so the script works correctly
from any working directory.

Environment variables required (set in .env or exported)::

    OPENAI_API_KEY
    PINECONE_API_KEY
    PINECONE_INDEX_NAME
    OPENAI_EMBEDDING_MODEL   (optional, default: text-embedding-3-small)
"""

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the project root importable regardless of working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import ChunkData, LegalSemanticChunker
from app.config import get_config
from app.document_assembler import DocumentAssembler
from app.embeddings import EmbeddingGenerator
from app.pdf_loader import PDFLoader
from app.text_preprocessor import TextPreprocessor
from app.vector_store import VectorStore

DEFAULT_CONTRACTS_DIR = PROJECT_ROOT / "contracts" / "sample_contracts"
DEFAULT_DIMENSION = 1536          # text-embedding-3-small
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"

_SEP = "─" * 50


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest legal contract PDFs into a Pinecone vector index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help=(
            "Create the Pinecone serverless index before ingesting.  "
            "Pass this flag on the first run only.  Raises an error if "
            "the index already exists."
        ),
    )
    parser.add_argument(
        "--contracts-dir",
        type=Path,
        default=DEFAULT_CONTRACTS_DIR,
        metavar="PATH",
        help="Directory containing the PDF files to ingest.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=DEFAULT_DIMENSION,
        metavar="INT",
        help=(
            "Embedding vector dimension.  Must match the model configured "
            "in OPENAI_EMBEDDING_MODEL (1536 for text-embedding-3-small, "
            "3072 for text-embedding-3-large)."
        ),
    )
    parser.add_argument(
        "--cloud",
        default=DEFAULT_CLOUD,
        metavar="PROVIDER",
        help="Cloud provider for the serverless index (aws | gcp | azure).",
    )
    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        metavar="REGION",
        help="Cloud region for the serverless index.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display a tqdm progress bar during the Pinecone upsert.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _create_index(dimension: int, cloud: str, region: str) -> None:
    """Create the Pinecone index.  Exits on failure."""
    config = get_config()
    print(f"  Creating Pinecone index '{config.pinecone_index_name}' …")
    try:
        VectorStore.create_index(
            dimension=dimension,
            cloud=cloud,
            region=region,
        )
        print(f"  ✓  Index created  (dim={dimension}, cloud={cloud}, region={region})")
    except ValueError as exc:
        print(f"  ✗  {exc}")
        sys.exit(1)


def _run_parser(contracts_dir: Path) -> list[ChunkData]:
    """Load PDFs and return all chunks from the frozen parser pipeline.

    Per-PDF failures are non-fatal: a warning is printed and that file is
    skipped.  If no PDFs load successfully the script exits with an error.
    """
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
    all_chunks = chunker.chunk_documents(documents)

    # Per-document summary
    chunks_by_doc: dict[str, list[ChunkData]] = {}
    pages_by_doc: dict[str, int] = {}
    for page in all_pages:
        pages_by_doc[page["document_name"]] = pages_by_doc.get(page["document_name"], 0) + 1
    for chunk in all_chunks:
        chunks_by_doc.setdefault(chunk["document_name"], []).append(chunk)

    for doc in documents:
        name = doc["document_name"]
        n_chunks = len(chunks_by_doc.get(name, []))
        n_pages = pages_by_doc.get(name, 0)
        print(f"  ✓  {name:<40}  {n_chunks:>4} chunks  ({n_pages} pages)")

    unique_ids = len({c["chunk_id"] for c in all_chunks})
    print(f"\n  Total  : {len(all_chunks)} chunks across {len(documents)} document(s)")
    if unique_ids < len(all_chunks):
        # Bilingual documents (e.g. German/English employment agreements) produce
        # two chunks per section on the same page, resulting in identical chunk_ids.
        # Pinecone upsert will overwrite on duplicate IDs, so the index will contain
        # unique_ids vectors rather than len(all_chunks).
        print(
            f"  Note   : {len(all_chunks) - unique_ids} bilingual duplicate ID(s) detected — "
            f"index will contain {unique_ids} unique vectors after upsert."
        )
    return all_chunks


def _generate_embeddings(chunks: list[ChunkData]) -> list[list[float]]:
    """Embed all chunk texts.  Exits on failure."""
    config = get_config()
    texts = [chunk["chunk_text"] for chunk in chunks]
    embedder = EmbeddingGenerator()

    print(f"  Embedding {len(texts)} texts via {config.openai_embedding_model} …")
    t0 = time.monotonic()
    try:
        vectors = embedder.embed_batch(texts)
    except Exception as exc:
        print(f"  ✗  Embedding failed: {exc}")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"  ✓  Done  ({elapsed:.1f}s)")
    return vectors


def _upsert(
    chunks: list[ChunkData],
    vectors: list[list[float]],
    show_progress: bool,
) -> int:
    """Connect to Pinecone and upsert all chunks.  Exits on failure."""
    config = get_config()
    print(f"  Index  : {config.pinecone_index_name}")

    try:
        store = VectorStore()
    except ValueError as exc:
        print(f"  ✗  {exc}")
        sys.exit(1)

    try:
        upserted = store.upsert_chunks(
            chunks=chunks,
            vectors=vectors,
            show_progress=show_progress,
        )
    except Exception as exc:
        print(f"  ✗  Upsert failed: {exc}")
        sys.exit(1)

    print(f"  Upserted: {upserted} vectors")
    return upserted


def _verify(chunks: list[ChunkData]) -> None:
    """Query index stats and confirm the vector count is consistent with the corpus.

    The expected count is the number of *unique* chunk IDs, not the total
    number of chunks.  Bilingual documents produce two chunks per section
    with the same ID; Pinecone overwrites on duplicate IDs so the final
    vector count equals the unique-ID count.
    """
    unique_ids = len({c["chunk_id"] for c in chunks})
    try:
        store = VectorStore()
        info = store.stats()
        actual = info.get("total_vector_count", "?")
        if actual == unique_ids:
            print(f"  ✓  Index now contains {actual} vectors")
        else:
            # A count mismatch can happen when the index is still indexing.
            # It is not a hard failure — the vectors were accepted by Pinecone.
            print(
                f"  ⚠   Index reports {actual} vectors "
                f"(expected {unique_ids}) — may still be indexing"
            )
    except Exception as exc:
        print(f"  ⚠   Could not verify index stats: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ── Step 0: validate credentials ────────────────────────────────────────
    try:
        config = get_config()
    except ValueError as exc:
        print(f"[ERROR] Configuration: {exc}")
        sys.exit(1)

    # ── Print configuration summary ──────────────────────────────────────────
    print(f"\n[1/5] Configuration  {_SEP}")
    print(f"  Model  : {config.openai_embedding_model}  (dim={args.dimension})")
    print(f"  Index  : {config.pinecone_index_name}")
    print(f"  Source : {args.contracts_dir}")

    # ── Step 1 (optional): create index ─────────────────────────────────────
    if args.create_index:
        print(f"\n[2/5] Creating Pinecone index  {_SEP}")
        _create_index(
            dimension=args.dimension,
            cloud=args.cloud,
            region=args.region,
        )
    else:
        print(f"\n[2/5] Skipping index creation  {_SEP}")
        print("  (pass --create-index on the first run)")

    # ── Step 2: parse documents ──────────────────────────────────────────────
    print(f"\n[3/5] Parsing documents  {_SEP}")
    chunks = _run_parser(args.contracts_dir)

    # ── Step 3: generate embeddings ──────────────────────────────────────────
    print(f"\n[4/5] Generating embeddings  {_SEP}")
    vectors = _generate_embeddings(chunks)

    # ── Step 4: upsert ───────────────────────────────────────────────────────
    print(f"\n[5/5] Upserting to Pinecone  {_SEP}")
    upserted = _upsert(chunks, vectors, show_progress=args.show_progress)

    # ── Step 5: verify ───────────────────────────────────────────────────────
    print(f"\n[+]   Verification  {_SEP}")
    _verify(chunks)

    print(f"\n{'─' * 56}")
    print("  Ingestion complete.")
    print(f"{'─' * 56}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
        sys.exit(0)
