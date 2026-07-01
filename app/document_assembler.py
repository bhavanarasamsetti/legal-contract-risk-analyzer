from collections import defaultdict
from typing import TypedDict

from app.pdf_loader import PageData


class DocumentData(TypedDict):
    """Typed structure representing a complete document and all its pages.

    Attributes:
        document_name: Filename of the source PDF (e.g. ``nda.pdf``).
        pages: Ordered list of :class:`~app.pdf_loader.PageData` dicts,
            one per non-empty page, sorted ascending by page number.
        page_count: Total number of pages in this document after cleaning.
            Equivalent to ``len(pages)`` but stored explicitly for convenient
            access without dereferencing the pages list.
    """

    document_name: str
    pages: list[PageData]
    page_count: int


class DocumentAssembler:
    """Groups a flat list of pages into per-document collections.

    After :class:`~app.pdf_loader.PDFLoader` extracts pages and
    :class:`~app.text_preprocessor.TextPreprocessor` cleans them, the result
    is a flat ``list[PageData]`` that may interleave pages from multiple PDFs.
    ``DocumentAssembler`` partitions that list into one :class:`DocumentData`
    per unique ``document_name``, restoring the natural document boundary.

    No text is modified, merged, or removed. Every page's metadata
    (``document_name``, ``page``) and ``text`` are passed through unchanged.

    Example:
        >>> assembler = DocumentAssembler()
        >>> documents = assembler.assemble_documents(clean_pages)
        >>> for doc in documents:
        ...     print(doc["document_name"], "—", doc["page_count"], "pages")
    """

    def assemble_documents(self, pages: list[PageData]) -> list[DocumentData]:
        """Group a flat list of pages into per-document structures.

        Pages that share the same ``document_name`` are collected into a single
        :class:`DocumentData`. Within each document, pages are explicitly sorted
        by their ``page`` number so correct order is guaranteed regardless of the
        order pages arrive in the input list.

        One :class:`DocumentData` is returned per unique ``document_name``.
        Documents appear in the output in the order their first page is
        encountered in the input list.

        Args:
            pages: Flat list of :class:`~app.pdf_loader.PageData` dicts.
                Typically the output of
                :class:`~app.text_preprocessor.TextPreprocessor`.

        Returns:
            A list of :class:`DocumentData` dicts, one per unique
            ``document_name``, each with pages sorted ascending by page number
            and ``page_count`` set to the number of pages in that document.
            Returns an empty list when ``pages`` is empty.

        Example:
            >>> assembler = DocumentAssembler()
            >>> pages = [
            ...     {"document_name": "nda.pdf", "page": 1, "text": "..."},
            ...     {"document_name": "nda.pdf", "page": 2, "text": "..."},
            ...     {"document_name": "dpa.pdf", "page": 1, "text": "..."},
            ... ]
            >>> docs = assembler.assemble_documents(pages)
            >>> [(d["document_name"], d["page_count"]) for d in docs]
            [('nda.pdf', 2), ('dpa.pdf', 1)]
        """
        # Use an insertion-ordered mapping so documents appear in first-seen order
        grouped: dict[str, list[PageData]] = defaultdict(list)

        for page in pages:
            grouped[page["document_name"]].append(page)

        documents: list[DocumentData] = []

        for name, doc_pages in grouped.items():
            # Guarantee ascending page order regardless of ingestion order
            sorted_pages = sorted(doc_pages, key=lambda page: page["page"])

            documents.append(
                DocumentData(
                    document_name=name,
                    pages=sorted_pages,
                    page_count=len(sorted_pages),
                )
            )

        return documents
