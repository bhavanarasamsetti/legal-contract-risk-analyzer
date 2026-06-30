from pathlib import Path

from app.pdf_loader import PageData, PDFLoader

# Directory that holds the sample contracts to ingest
CONTRACTS_DIR = Path(__file__).resolve().parent.parent / "contracts" / "sample_contracts"

# Number of characters to show in the page preview
PREVIEW_LENGTH = 200

# Visual separators
DOC_SEPARATOR = "-" * 50
SECTION_SEPARATOR = "=" * 46


def print_document_summary(document_name: str, pages: list[PageData]) -> None:
    """Print a formatted ingestion summary for a single PDF document.

    Args:
        document_name: Filename of the PDF (e.g. ``employment_agreement.pdf``).
        pages: List of extracted page dicts returned by :class:`PDFLoader`.
    """
    print(DOC_SEPARATOR)
    print(f"Document: {document_name}")
    print(f"Pages Extracted: {len(pages)}")
    print(DOC_SEPARATOR)

    for page_data in pages:
        preview = page_data["text"][:PREVIEW_LENGTH]

        print(f"\nPage {page_data['page']}")
        print(f"Characters: {len(page_data['text'])}")
        print("\nPreview:")
        print(preview)
        print()
        print(DOC_SEPARATOR)


def main() -> None:
    """Discover every PDF in CONTRACTS_DIR, load it, and print an ingestion summary.

    Iterates over all ``.pdf`` files in the contracts directory, loads each one
    with a single shared :class:`PDFLoader` instance, and prints a page-level
    summary so the ingestion pipeline can be manually verified.
    """
    # Collect and sort PDF paths for deterministic output
    pdf_files = sorted(CONTRACTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {CONTRACTS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s) in '{CONTRACTS_DIR}'\n")

    # One loader instance is sufficient — PDFLoader is stateless
    loader = PDFLoader()

    for pdf_path in pdf_files:
        try:
            pages = loader.load_pdf(str(pdf_path))
            print_document_summary(pdf_path.name, pages)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"[ERROR] Could not load '{pdf_path.name}': {exc}")

        print(SECTION_SEPARATOR)


if __name__ == "__main__":
    main()
