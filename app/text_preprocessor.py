import re

from app.pdf_loader import PageData

_PARAGRAPH_SENTINEL = "<<PARAGRAPH>>"

# Matches standalone page-number lines that PDF extraction leaves behind.
# Every alternative is anchored at both ends (^ … $) with re.MULTILINE so it
# only fires when the ENTIRE line is a page-number expression — never when a
# number appears as the opening token of a real legal heading such as
# "1. Definitions" or "§ 12 Obligations".
#
# Patterns handled:
#   "1 / 16"       — N / M
#   "2 / 16"       — same
#   "11"           — bare integer on its own line
#   "24"           — same
#   "Page 3 of 24" — "Page N of M"
#   "3 of 24"      — "N of M"
_PAGE_ARTIFACT_RE = re.compile(
    r"^\d+\s*/\s*\d+$"               # "1 / 16"
    r"|^[Pp]age\s+\d+\s+of\s+\d+$"  # "Page 3 of 24"
    r"|^\d+\s+of\s+\d+$"             # "3 of 24"
    r"|^\d+$",                        # bare integer: "11", "24"
    re.MULTILINE,
)
# Matches page numbers merged with the first clause on the same line.
# Example:
#   "16 / 16 12.5 Governing Law"
# becomes:
#   "12.5 Governing Law"
_INLINE_PAGE_PREFIX_RE = re.compile(
    r"^\d+\s*/\s*\d+\s+"
)

class TextPreprocessor:
    """Cleans and normalises raw PDF text to prepare it for semantic chunking.

    This class operates exclusively on the text field of each :class:`PageData`
    dict. All document metadata (``document_name``, ``page``) is preserved
    unchanged. No chunking, embedding, or downstream processing is performed.

    The cleaning pipeline applied to every page is:

    1. :meth:`_remove_page_headers_footers` — strip standalone page-number
       lines such as ``"7 / 16"`` or ``"24"`` that PDF extraction leaves
       behind.
    2. :meth:`_fix_line_breaks`  — merge broken lines inside sentences while
       keeping genuine paragraph boundaries.
    3. :meth:`_clean_text`       — remove non-printable / control characters
       that pypdf sometimes emits.
    4. :meth:`_normalize_whitespace` — collapse runs of spaces and tabs; strip
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
            text = self._remove_page_headers_footers(text)
            text = self._remove_inline_page_prefixes(text)
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

    def _remove_page_headers_footers(self, text: str) -> str:
        """Remove standalone page-number lines left behind by PDF extraction.

        Many PDFs embed a running header or footer (e.g. ``"7 / 16"``,
        ``"Page 3 of 24"``, or a bare page number ``"11"``) on every page.
        After extraction these appear as isolated lines in the text.  They
        carry no legal meaning and must be stripped before sentence-level
        processing so they do not corrupt line-break repair or chunking.

        Only lines whose *entire content* matches a page-number pattern are
        removed.  Lines where a number is followed by legal text — such as
        ``"1. Definitions"``, ``"§ 12 Obligations"``, or
        ``"ARTICLE III Termination"`` — are left completely untouched.

        Args:
            text: Raw text from a single PDF page.

        Returns:
            Text with standalone page-header and page-footer lines removed.
            Surrounding whitespace is not otherwise altered at this stage.

        Example:
            >>> preprocessor = TextPreprocessor()
            >>> preprocessor._remove_page_headers_footers("7 / 16\\nSome clause text.")
            'Some clause text.'
            >>> preprocessor._remove_page_headers_footers("1. Definitions\\nText here.")
            '1. Definitions\\nText here.'
        """
        # Replace each matching line with an empty string; _normalize_whitespace
        # will later collapse the resulting blank lines.
        return _PAGE_ARTIFACT_RE.sub("", text)

    def _remove_inline_page_prefixes(self, text: str) -> str:
        """Remove merged page-number prefixes."""

        cleaned_lines = []

        for line in text.splitlines():
            cleaned_lines.append(
                _INLINE_PAGE_PREFIX_RE.sub("", line)
            )

        return "\n".join(cleaned_lines)

    def _fix_line_breaks(self, text: str) -> str:
        """Merge soft-wrapped lines while preserving legal clause boundaries.

        PDF extraction wraps long lines with a single ``\\n`` that carries no
        structural meaning.  However, legal documents also use single newlines
        to separate numbered clauses on consecutive lines (e.g. ``12.5.`` and
        ``12.6.`` each on their own line).  This method distinguishes between
        the two cases:

        - **Soft-wrap** — a line continues the previous sentence.  The ``\\n``
          is replaced with a space so the two fragments read as one sentence.
        - **Clause boundary** — the next line opens a new legal clause
          (detected by its heading token).  A blank line is inserted before it
          so the separator becomes ``\\n\\n``, which the downstream chunker
          uses as a paragraph boundary via ``text.split("\\n\\n")``.

        Strategy:
            1. Protect genuine paragraph breaks (``\\n\\n`` or more) with a
               sentinel so they survive the line-by-line scan.
            2. Iterate over every line.  If the line starts with a recognised
               legal heading token, insert a blank line before it (promoting
               the boundary to ``\\n\\n``) and keep the heading on its own
               line.  Otherwise merge the line with the previous one (soft-wrap).
            3. Restore the paragraph breaks from the sentinel.

        Args:
            text: Raw text from a single PDF page, after page-artifact removal.

        Returns:
            Text where soft-wrapped continuations are joined and clause-level
            ``\\n`` boundaries are preserved so that ``text.split("\\n\\n")``
            still produces meaningful legal paragraphs.
        """
        # 1. Protect genuine paragraph breaks (2+ newlines) with a sentinel
        text = re.sub(r"\n{2,}", _PARAGRAPH_SENTINEL, text)

        # Recognises the opening token of a legal heading so the newline
        # before it is kept as a clause boundary rather than collapsed.
        # Covers: §1 / § 12 / 1. / 2.1 / 3.4.2 / Section 5 / Article III /
        #         PART IV / Clause 8 / Schedule A / Annex 1 / Exhibit B
        _heading_start = re.compile(
            r"^(?:"
            r"§\s*\d+"                                                     # §1 / § 12
            r"|[0-9]+(?:\.[0-9]+)*\."                                      # 1. / 2.1. / 3.4.2.
            r"|[0-9]+(?:\.[0-9]+)+"                                        # 2.1 / 3.4.2 (no trailing dot)
            r"|(?:Section|Article|Clause|Schedule|Exhibit|Annex|Appendix|Part)\s"  # keyword headings
            r")",
            re.IGNORECASE,
        )

        # 2. Rebuild text line-by-line, merging soft-wraps and keeping
        #    clause-boundary newlines intact.
        segments = text.split("\n")
        output: list[str] = [segments[0]] if segments else []

        for segment in segments[1:]:
            stripped = segment.strip()

            # Paragraph sentinels must never be merged into the previous line
            if _PARAGRAPH_SENTINEL in segment:
                output.append(segment)
                continue

            # Blank lines are kept; _normalize_whitespace tidies them later
            if not stripped:
                output.append(segment)
                continue

            # A line that opens a new legal clause is a paragraph boundary.
            # Insert a blank line before it so the separator becomes \n\n,
            # which lets the chunker split correctly via text.split("\n\n").
            if _heading_start.match(stripped):
                if output and output[-1].strip():  # avoid duplicate blank lines
                    output.append("")
                output.append(segment)
                continue

            # Soft-wrap: join this continuation with the previous line
            if output and output[-1].strip() and _PARAGRAPH_SENTINEL not in output[-1]:
                output[-1] = output[-1].rstrip() + " " + segment.lstrip()
            else:
                output.append(segment)

        text = "\n".join(output)

        # 3. Restore paragraph breaks from the sentinel
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
            - Blank lines that contain only whitespace → proper paragraph break.
            - Spaces immediately before or after a paragraph break (``\\n\\n``).
            - Leading and trailing whitespace on the full string.

        Args:
            text: Text after control-character removal.

        Returns:
            Whitespace-normalised text.
        """
        # Collapse runs of spaces/tabs (but not newlines) into a single space
        text = re.sub(r"[ \t]+", " ", text)

        # Normalise blank lines that contain only whitespace into a proper
        # paragraph break.  Multi-column PDFs (e.g. bilingual two-column
        # employment agreements) produce lines like "\n    \n" — a newline,
        # spaces for the column gutter, then another newline — instead of the
        # "\n\n" the chunker expects.  After the space-collapse step above such
        # lines become "\n \n"; this substitution converts them to "\n\n" so
        # that text.split("\n\n") in the chunker correctly separates paragraphs.
        # Single-column documents (Atlassian DPA, Data Processing Agreement)
        # already use "\n\n" for paragraph breaks, so this substitution makes
        # zero replacements on those documents.
        text = re.sub(r"\n[ \t]+\n", "\n\n", text)

        # Remove spaces that crept in around paragraph breaks
        text = re.sub(r" *\n\n *", "\n\n", text)

        return text.strip()
