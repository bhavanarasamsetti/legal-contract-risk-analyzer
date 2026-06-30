from pathlib import Path
from typing import TypedDict

import pypdf


class PageData(TypedDict):
    """Typed structure for a single extracted PDF page."""

    document_name: str
    page: int
    text: str


class PDFLoader:
    """Handles loading and extracting text content from PDF files.

    This class is responsible solely for the PDF ingestion layer.
    It performs file validation, reads page content using pypdf, and
    returns a clean, structured list of non-empty pages.

    Example:
        >>> loader = PDFLoader()
        >>> pages = loader.load_pdf("contract.pdf")
        >>> print(pages[0]["text"])
    """

    def load_pdf(self, file_path: str) -> list[PageData]:
        """Load a PDF file and extract text from each non-empty page.

        Args:
            file_path: Absolute or relative path to the PDF file.

        Returns:
            A list of dicts, one per non-empty page, each containing:
                - ``document_name`` (str): Filename of the source PDF.
                - ``page`` (int): 1-based page number.
                - ``text`` (str): Extracted text content of the page.

        Raises:
            FileNotFoundError: If no file exists at ``file_path``.
            ValueError: If the file is not a ``.pdf`` file.
            RuntimeError: If pypdf fails to read or parse the file.

        Example:
            >>> loader = PDFLoader()
            >>> pages = loader.load_pdf("agreements/nda.pdf")
            >>> for p in pages:
            ...     print(f"Page {p['page']}: {len(p['text'])} chars")
        """
        self._validate_path(file_path)

        try:
            return self._extract_pages(file_path)
        except (pypdf.errors.PdfReadError, Exception) as exc:
            raise RuntimeError(
                f"Failed to read PDF '{file_path}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_path(self, file_path: str) -> None:
        """Ensure the file exists and carries a .pdf extension.

        Args:
            file_path: Path to validate.

        Raises:
            FileNotFoundError: If the path does not point to an existing file.
            ValueError: If the file extension is not ``.pdf``.
        """
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"PDF file not found: '{file_path}'. "
                "Please provide a valid path to an existing file."
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                f"Invalid file type '{path.suffix}'. Expected a '.pdf' file, got: '{file_path}'."
            )

    def _extract_pages(self, file_path: str) -> list[PageData]:
        """Open the PDF and extract text from each non-empty page.

        Args:
            file_path: Validated path to the PDF file.

        Returns:
            A list of :class:`PageData` dicts for all pages with content.
        """
        pages: list[PageData] = []
        document_name = Path(file_path).name

        with open(file_path, "rb") as fh:
            reader = pypdf.PdfReader(fh)

            for index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()

                # Skip pages that yield no usable text (e.g. image-only pages)
                if not text:
                    continue

                pages.append(
                    PageData(document_name=document_name, page=index + 1, text=text)
                )

        return pages
