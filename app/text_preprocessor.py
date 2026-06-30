import re

from app.pdf_loader import PageData

_PARAGRAPH_SENTINEL = "<<PARAGRAPH>>"


class TextPreprocessor:
    """Cleans and normalises raw PDF text to prepare it for semantic chunking.

    This class operates exclusively on the text field of each :class:`PageData`
    dict. All document metadata (``document_name``, ``page``) is preserved
    unchanged. No chunking, embedding, or downstream processing is performed.

    The cleaning pipeline applied to every page is:

    1. :meth:`_fix_line_breaks`  — merge broken lines inside sentences while
       keeping genuine paragraph boundaries.
    2. :meth:`_clean_text`       — remove non-printable / control characters
       that pypdf sometimes emits.
    3. :meth:`_normalize_whitespace` — collapse runs of spaces and tabs; strip
       leading/trailing whitespace.

    Example:
        >>> preprocessor = TextPreprocessor()
        >>> clean_pages = preprocessor.preprocess_pages(raw_pages)
    """

    def preprocess_pages(self, pages: list[PageData]) -> list[PageData]:
        """Apply the full cleaning pipeline to every page in the list.

        Each page's ``document_name`` and ``page`` number are copied verbatim;
        only the ``text`` field is transformed.

        Args:
            pages: Raw :class:`PageData` dicts as returned by
                :class:`~app.pdf_loader.PDFLoader`.

        Returns:
            A new list of :class:`PageData` dicts with cleaned ``text`` values.
            Pages whose text becomes empty after cleaning are dropped.

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> pages = [{"document_name": "nda.pdf", "page": 1, "text": "foo  bar\\n\\nbaz"}]
            >>> preprocessor.preprocess_pages(pages)
            [{'document_name': 'nda.pdf', 'page': 1, 'text': 'foo bar\\n\\nbaz'}]
        """
        cleaned: list[PageData] = []

        for page in pages:
            text = page["text"]
            text = self._fix_line_breaks(text)
            text = self._clean_text(text)
            text = self._normalize_whitespace(text)

            # Drop pages that are empty after cleaning (e.g. pure header pages)
            if not text:
                continue

            cleaned.append(
                PageData(
                    document_name=page["document_name"],
                    page=page["page"],
                    text=text,
                )
            )

        return cleaned

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fix_line_breaks(self, text: str) -> str:
        """Merge mid-sentence line breaks while preserving paragraph boundaries.

        PDF extraction often wraps long lines with a single newline (``\\n``)
        that is not a real paragraph boundary. Two or more consecutive newlines
        (``\\n\\n``) reliably indicate a clause or paragraph break in legal
        documents and must be kept intact.

        Strategy:
            1. Temporarily replace double-newlines with a sentinel so they
               survive the next step.
            2. Replace remaining single newlines between word characters with
               a space (i.e. re-join wrapped lines).
            3. Restore the paragraph breaks from the sentinel.

        Args:
            text: Raw text from a single PDF page.

        Returns:
            Text with mid-sentence line breaks joined and paragraph breaks
            preserved.
        """
        # Protect genuine paragraph breaks (2+ newlines) with a sentinel
        text = re.sub(r"\n{2,}", _PARAGRAPH_SENTINEL, text)

        # Join lines that were soft-wrapped mid-sentence:
        # only replace a single \n when it sits between word characters
        # or punctuation so we don't accidentally merge bullet points that
        # start with digits or letters after a clean break.
        text = re.sub(r"(?<=\S)\n(?=\S)", " ", text)

        # Any remaining lone newlines (e.g. after punctuation) can also be
        # treated as spaces — they are not structural breaks.
        text = text.replace("\n", " ")

        # Restore paragraph breaks as double newlines
        text = text.replace(_PARAGRAPH_SENTINEL, "\n\n")

        return text

    def _clean_text(self, text: str) -> str:
        """Remove non-printable and control characters left by PDF extraction.

        pypdf occasionally emits null bytes, form-feed characters, zero-width
        spaces, and other Unicode control characters that are invisible but
        corrupt downstream NLP pipelines.

        Keeps:
            - Standard printable ASCII and Unicode letters/punctuation.
            - Newlines (``\\n``) — already managed by :meth:`_fix_line_breaks`.
            - Spaces and regular horizontal whitespace.

        Args:
            text: Text after line-break normalisation.

        Returns:
            Text with all control characters (except ``\\n``) removed.
        """
        # Remove null bytes and form-feed characters explicitly
        text = text.replace("\x00", "").replace("\x0c", "")

        # Remove remaining ASCII control characters (0x00–0x1F) except \n (0x0A)
        text = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", "", text)

        # Remove Unicode private-use / specials that sometimes appear in PDFs
        text = re.sub(r"[\ufff0-\uffff]", "", text)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse redundant spaces and strip leading/trailing whitespace.

        Handles:
            - Multiple consecutive spaces or tabs → single space.
            - Spaces immediately before or after a paragraph sentinel (``\\n\\n``).
            - Leading and trailing whitespace on the full string.

        Args:
            text: Text after control-character removal.

        Returns:
            Whitespace-normalised text.
        """
        # Collapse runs of spaces/tabs (but not newlines) into a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Remove spaces that crept in around paragraph breaks
        text = re.sub(r" *\n\n *", "\n\n", text)

        return text.strip()
