import re
from typing import TypedDict

from app.document_assembler import DocumentData

# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Roman numeral component used inside heading patterns below.
# Matches standard uppercase Roman numerals I–XXXIX (covers all realistic
# legal document part/article counts).
# ---------------------------------------------------------------------------
_ROMAN = r"(?:X{0,3}(?:IX|IV|V?I{0,3}))"

# Page-header / page-footer patterns that must be rejected BEFORE the heading
# regex runs.  These lines look like a numeric heading but are not legal clauses:
#
#   "7 / 16"     — N / M  (page number / total pages)
#   "1 / 16"     — same
#   "11"         — bare page number with no following text
#   "24"         — same
#   "Page 3 of 24" — "Page N of M"
#   "3 of 24"    — "N of M"
#
# Every alternative is anchored at BOTH ends so it only fires when the entire
# first line is a page-number expression (never a real heading with a title).
_PAGE_HEADER_RE = re.compile(
    r"^\d+\s*/\s*\d+$"              # "7 / 16"
    r"|^[Pp]age\s+\d+\s+of\s+\d+$"  # "Page 3 of 24"
    r"|^\d+\s+of\s+\d+$"            # "3 of 24"
    r"|^\d+$",                       # bare integer alone on a line: "11", "24"
)

# Matches the opening of a legal section heading at the start of a line:
#
#   Keyword-based  : "Section 5"  "Article 7"  "Clause 9"  "ARTICLE III"
#                    "Schedule A"  "Exhibit B"  "Annex 1"
#   European (§)   : "§1"  "§ 1"  "§12"  "§ 12"
#   Numeric        : "1."  "2.1"  "3.4.2"  (bare integers filtered by
#                    _PAGE_HEADER_RE before this regex runs)
#   Lettered       : "A."  "B."  "IX."
#
# Keyword headings now also accept a trailing Roman numeral qualifier so that
# "ARTICLE I", "ARTICLE II", "PART IV" etc. are recognised.
_HEADING_RE = re.compile(
    r"^(?:"
    # Keyword headings — numeric, alpha, or Roman numeral qualifier
    r"(?:Section|Article|Clause|Schedule|Exhibit|Annex|Appendix|Part)"
    r"(?:\s+(?:[A-Z0-9]+(?:\.[0-9]+)*|" + _ROMAN + r"))?"
    # European section symbol: §1 / § 1 / §12 / § 12
    r"|§\s*[0-9]+"
    # Dotted or plain numeric: 1. / 2.1 / 3.4.2
    r"|(?:[0-9]+\.[0-9]+(?:\.[0-9]+)*|[0-9]+\.)"
    # Lettered: A. / B. / IX.
    r"|[A-Z]{1,3}\."
    r")\s*",
    re.IGNORECASE,
)

# Captures the descriptive title that follows the heading token on the same
# line, e.g. "3.2 Confidentiality Obligations" → title = "Confidentiality Obligations"
_HEADING_WITH_TITLE_RE = re.compile(
    r"^(?:"
    r"(?:Section|Article|Clause|Schedule|Exhibit|Annex|Appendix|Part)"
    r"(?:\s+(?:[A-Z0-9]+(?:\.[0-9]+)*|" + _ROMAN + r"))?"
    r"|§\s*[0-9]+"
    r"|[0-9]+(?:\.[0-9]+)*\.?"
    r"|[A-Z]{1,3}\."
    r")\s+(.+)$",
    re.IGNORECASE,
)

# Extracts the numeric depth of a heading token to infer parent/child
# relationships, e.g. "2.1" → depth 2, "3" → depth 1, "Section 5" → depth 1
_NUMERIC_HEADING_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)")

# Markers that indicate a line is a legal citation embedded in body text,
# not a standalone clause heading.  These terms appear in cross-references
# (e.g. "section 75 para. 2 GmbHG") but never in standalone clause titles.
_CITATION_MARKER_RE = re.compile(r"\bpara\.\s*\d|\bAbs\.\s*\d|\blit\.\s*[a-z]", re.IGNORECASE)

# Running page-header pattern specific to the Atlassian Customer DPA.
# Every page of that document starts with a line of the form:
#   "N Atlassian Customer DPA v.MM/DD/YY"
# These lines survive preprocessing because they are not bare page numbers,
# and they contaminate chunk text when they appear at the start of a
# paragraph.  The pattern matches that prefix so it can be stripped before
# the paragraph is added to any chunk.
_RUNNING_HEADER_RE = re.compile(
    r"^\d+\s+Atlassian Customer DPA\s+v\.\d{2}/\d{2}/\d{2}\s*",
    re.IGNORECASE,
)


class ChunkData(TypedDict):
    """Typed structure for a single semantic legal chunk.

    Attributes:
        chunk_id: Unique identifier derived from document name, first page
            number, and section token (e.g. ``"nda_pdf_1_3.2"``).
        document_name: Filename of the source PDF (e.g. ``nda.pdf``).
        pages: Sorted list of all 1-based page numbers that contribute text
            to this chunk. A chunk that spans pages 3 and 4 will have
            ``pages = [3, 4]``.
        section: The raw heading token that opened this chunk
            (e.g. ``"3.2"``). ``"preamble"`` when the text precedes any
            detected heading.
        section_title: The descriptive title following the heading token
            (e.g. ``"Confidentiality Obligations"``). Empty string when the
            heading carries no title.
        parent_section: The heading token of the nearest ancestor section
            (e.g. ``"3"`` for a chunk with ``section="3.2"``). Empty string
            for top-level sections and the preamble.
        chunk_text: Full text of the clause or subsection, including the
            heading line, with paragraphs separated by ``\\n\\n``.
    """

    chunk_id: str
    document_name: str
    pages: list[int]
    section: str
    section_title: str
    parent_section: str
    chunk_text: str


class LegalSemanticChunker:
    """Converts assembled legal documents into semantically meaningful chunks.

    Each chunk represents one complete legal clause or subsection delimited
    by a recognised section heading. Chunking never crosses a heading
    boundary and is never driven by character count.

    Key behaviours:

    - Processes one complete :class:`~app.document_assembler.DocumentData` at
      a time, so clause text that spans multiple pages stays in a single chunk.
    - Tracks which pages contributed to each chunk via the ``pages`` field.
    - Infers a ``parent_section`` from the numeric depth of the heading token
      (e.g. section ``"3.2"`` is a child of ``"3"``).
    - Generates a deterministic ``chunk_id`` from the document name, the first
      page of the chunk, and the section token.

    Example:
        >>> chunker = LegalSemanticChunker()
        >>> chunks = chunker.chunk_documents(documents)
        >>> for c in chunks:
        ...     print(c["chunk_id"], "—", c["section_title"])
    """

    def chunk_documents(self, documents: list[DocumentData]) -> list[ChunkData]:
        """Segment a list of assembled documents into semantic legal chunks.

        For each document, pages are iterated in order and their text is split
        on paragraph boundaries (``\\n\\n``). Paragraphs are accumulated under
        the nearest preceding heading; when a new heading is encountered the
        current accumulator is flushed as a :class:`ChunkData`. Text before
        the first heading is emitted as a ``"preamble"`` chunk.

        Because pages are consumed sequentially across the full document (not
        per-page), a clause that begins on page 4 and continues on page 5 is
        returned as a single chunk with ``pages = [4, 5]``.

        Args:
            documents: List of :class:`~app.document_assembler.DocumentData`
                dicts, typically produced by
                :class:`~app.document_assembler.DocumentAssembler`.

        Returns:
            A flat list of :class:`ChunkData` dicts ordered by document then
            by appearance in the source text. Returns an empty list when
            ``documents`` is empty.

        Example:
            >>> chunker = LegalSemanticChunker()
            >>> chunks = chunker.chunk_documents(documents)
            >>> print(chunks[0]["chunk_id"])
            'nda_pdf_1_preamble'
        """
        chunks: list[ChunkData] = []

        for document in documents:
            chunks.extend(self._chunk_document(document))

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_document(self, document: DocumentData) -> list[ChunkData]:
        """Chunk a single document across all its pages.

        Pages are processed in their stored order (already sorted by
        :class:`~app.document_assembler.DocumentAssembler`). Paragraphs from
        consecutive pages are fed into the same accumulator so that a legal
        clause is never split at a page boundary.

        Args:
            document: A single assembled document.

        Returns:
            Ordered list of :class:`ChunkData` dicts for this document.
        """
        chunks: list[ChunkData] = []
        document_name = document["document_name"]

        # Running accumulator state
        current_section: str = "preamble"
        current_title: str = ""
        current_parent: str = ""
        current_paragraphs: list[str] = []
        current_pages: list[int] = []

        for page_data in document["pages"]:
            page_number = page_data["page"]
            paragraphs = [p.strip() for p in page_data["text"].split("\n\n")]
            paragraphs = [p for p in paragraphs if p]

            # Strip running page-headers (e.g. "6 Atlassian Customer DPA v.07/30/23")
            # from the start of every paragraph.  These headers were not caught by
            # the preprocessor because they contain document-title text, not a bare
            # page number.  After stripping, discard any paragraph that becomes empty
            # (the header was the only content on that line).
            cleaned: list[str] = []
            for p in paragraphs:
                p = _RUNNING_HEADER_RE.sub("", p).strip()
                if p:
                    cleaned.append(p)
            paragraphs = cleaned

            print("\n" + "=" * 80)
            print(f"DOCUMENT: {document_name}")
            print(f"PAGE: {page_number}")
            print(f"PARAGRAPHS FOUND: {len(paragraphs)}")
            print("=" * 80)

            for i, paragraph in enumerate(paragraphs, start=1):
                print(f"\n--- Paragraph {i} ---")
                print(repr(paragraph[:300]))

            for paragraph in paragraphs:
                heading = self._extract_section_heading(paragraph)

                if heading is not None:
                    # Flush the current accumulator before starting a new section
                    if current_paragraphs:
                        chunks.append(
                            self._build_chunk(
                                document_name=document_name,
                                pages=current_pages,
                                section=current_section,
                                section_title=current_title,
                                parent_section=current_parent,
                                paragraphs=current_paragraphs,
                            )
                        )

                    section_token, section_title = heading
                    current_section = section_token
                    current_title = section_title
                    current_parent = self._infer_parent_section(section_token)
                    current_paragraphs = [paragraph]
                    current_pages = [page_number]
                else:
                    current_paragraphs.append(paragraph)
                    # Track the page even if it was already added
                    if not current_pages or current_pages[-1] != page_number:
                        current_pages.append(page_number)

        # Flush whatever remains after the last page of the document
        if current_paragraphs:
            chunks.append(
                self._build_chunk(
                    document_name=document_name,
                    pages=current_pages,
                    section=current_section,
                    section_title=current_title,
                    parent_section=current_parent,
                    paragraphs=current_paragraphs,
                )
            )

        return chunks

    def _extract_section_heading(self, paragraph: str) -> tuple[str, str] | None:
        """Determine whether a paragraph contains a legal section heading.

        Scans the paragraph line by line, skipping blank lines and lines that
        match :data:`_PAGE_HEADER_RE`, until the first line that matches
        :data:`_HEADING_RE`.  The section token and title are extracted from
        that line only and returned immediately.  If no matching line is found,
        ``None`` is returned.

        This scan-based approach is more robust than inspecting only the first
        line because PDF extraction frequently prepends page numbers, page
        stamps, or blank lines before the real heading text.

        Args:
            paragraph: A single paragraph block (typically split on ``\\n\\n``).

        Returns:
            A ``(section_token, section_title)`` tuple when a heading is
            detected, or ``None`` when the paragraph contains no heading.

            - ``section_token`` — the raw heading token, e.g. ``"3.2"``,
              ``"§12"``, or ``"Article III"``.
            - ``section_title`` — the remainder of the heading line only,
              e.g. ``"Confidentiality Obligations"``; empty string when the
              heading carries no title text.

        Example:
            >>> chunker = LegalSemanticChunker()
            >>> chunker._extract_section_heading("11\\n3.2 Confidentiality\\nText...")
            ('3.2', 'Confidentiality')
            >>> chunker._extract_section_heading("7 / 16")
            >>> chunker._extract_section_heading("This agreement is entered into...")
        """
        for line in paragraph.splitlines():
            line = line.strip()

            # Skip blank lines
            if not line:
                continue

            # Skip standalone page-number / page-header lines
            if _PAGE_HEADER_RE.match(line):
                continue

            # First non-noise line that matches a heading pattern wins
            if _HEADING_RE.match(line):
                token_match = _HEADING_RE.match(line)
                section_token = token_match.group(0).strip()  # type: ignore[union-attr]

                # Reject tokens that consist entirely of lowercase letters
                # followed by a period (e.g. "ii.", "iii.", "iv.").  These are
                # ordered sub-list markers, not standalone section headings.
                # They can appear at the start of a paragraph when a page break
                # falls mid-list and the running page header is later stripped.
                if re.match(r"^[a-z]+\.$", section_token):
                    return None

                # --- Title extraction ---
                #
                # Derive the title from the remainder of the line after the
                # token match ends.  Using token_match.end() is always
                # consistent with the actual token boundary, whereas
                # _HEADING_WITH_TITLE_RE can backtrack and capture orphaned
                # text (e.g. the "(B)" from "Annex 1(B) Title") in the title.
                title_raw = line[token_match.end():].strip()

                # Strip leading artefacts that _HEADING_RE could not consume:
                #   1. A leftover period: "2.1." — the trailing dot belongs to
                #      the token but is not consumed by the dotted-number branch,
                #      leaving ". Title text" as the raw remainder.
                #   2. A parenthetical qualifier with an optional trailing colon
                #      or dash: "(B)" from "Annex 1(B) Title" or "(C):" from
                #      "Annex 1(C): Competent supervisory authority".  The colon
                #      is part of the heading format, not a content separator.
                title_raw = re.sub(r"^\.?\s*(?:\([A-Za-z]+\)\s*[:\-]?\s*)?", "", title_raw)

                # --- Sentence-body detection ---
                #
                # If the remainder of the line opens with a word that starts
                # a sentence rather than a heading noun phrase, the clause
                # body begins immediately after the section number and no
                # title is present.  Genuine heading titles are short noun
                # phrases; they never begin with articles, demonstratives,
                # personal pronouns, prepositions, or common legal-clause
                # openers such as "Notwithstanding" or "Except for".
                #
                # Examples that must produce section_title = "":
                #   "The parties agree that this DPA replaces …"     (the)
                #   "This DPA and the Standard Contractual Clauses …" (this)
                #   "Except for the changes made by this DPA …"      (except)
                #   "Notwithstanding anything to the contrary …"     (notwithstanding)
                #   "Any claims against Atlassian …"                 (any)
                _sentence_opener = re.compile(
                    r"^(?:the|this|these|that|those|a|an|it|its|"
                    r"each|any|all|no\b|neither|"
                    r"notwithstanding|except|unless|"
                    r"subject\s+to|pursuant\s+to|in\s+accordance|"
                    r"where|when|if|for|by|upon)\b",
                    re.IGNORECASE,
                )
                if _sentence_opener.match(title_raw):
                    section_title = ""
                else:
                    # Locate the earliest position where body prose begins.
                    # Legal heading titles are short noun phrases; everything
                    # that follows is body text.  We detect the boundary via
                    # the first match of any of these patterns:
                    #
                    #   \n               — newline separates heading from body
                    #   :\s              — colon introduces body ("Definitions: …")
                    #   \.\s             — period ends the heading phrase
                    #   ,\s(the|this|…)  — comma before a clause-intro article
                    #   \s(will|shall|…) — predicate modal/copula starts the
                    #                       main clause
                    #   )\s[A-Z]         — parenthetical qualifier ends and an
                    #                       uppercase sentence begins
                    _body_start = re.compile(
                        r"\n"
                        r"|:\s"
                        r"|\.\s"
                        r"|,\s+(?:the|this|a|an|that|which|where|unless)\b"
                        r"|\s+(?:will|shall|must|is|are|was|were|may|have|has)\b"
                        r"|\)\s+[A-Z]",
                        re.IGNORECASE,
                    )
                    _bm = _body_start.search(title_raw)
                    if _bm:
                        # When the match starts on ")" include it in the title
                        # (the closing paren belongs to the qualifier phrase).
                        _stop = _bm.start() + (1 if title_raw[_bm.start()] == ")" else 0)
                    else:
                        _stop = len(title_raw)

                    # Additional case-sensitive check: a capitalised determiner
                    # ("The …", "This …") that appears in the middle of the
                    # remainder is almost always the start of a new sentence, not
                    # part of the heading title.  Apply only when the word is
                    # preceded by at least one character so we do not fire at the
                    # very start of the title (which is already handled by the
                    # sentence-opener gate above).
                    _cap_m = re.search(r"\s+(?:The|This)\s+", title_raw)
                    if _cap_m and _cap_m.start() > 0:
                        _stop = min(_stop, _cap_m.start())

                    section_title = title_raw[:_stop].rstrip(".:,").strip()

                # --- Post-extraction validation ---

                # Reject lines that contain cross-reference citation markers
                # (e.g. "section 75 para. 2", "§ 12 Abs. 3") — these are
                # inline citations, not clause headings.
                if _CITATION_MARKER_RE.search(line):
                    return None

                # Reject inline sub-clause cross-references of the form
                # "Section 2.5(c) of this DPA", "Section 2.2(a) above", or
                # "1.4(i) and (ii) of this Agreement".
                #
                # The pattern  \d\s*\([a-z]\)  matches a digit immediately
                # followed by a parenthesised single lowercase letter — the
                # universal legal notation for sub-clause references.
                # Standalone headings (e.g. "Section 5 Liability") never
                # carry this notation; it appears exclusively in body text
                # that cites a specific numbered sub-clause.
                #
                # IMPORTANT: search only the first 50 characters of the line
                # (the heading token + immediate context).  Long paragraphs
                # that start with a valid heading but then contain self-
                # references in the body (e.g. "…see Section 2.14(a) above")
                # would otherwise be falsely rejected.  At 50 chars we cover
                # any realistic heading token while staying clear of body text.
                if re.search(r"\d\s*\([a-z]\)", line[:50]):
                    return None

                # Reject numeric tokens that use the European thousands-separator
                # style (e.g. "50.000" means 50,000 not section 50, sub-clause 0).
                # Real legal section numbers have at most 2 digits per component.
                numeric_match = _NUMERIC_HEADING_RE.match(section_token)
                if numeric_match:
                    components = numeric_match.group(1).split(".")
                    if any(len(c) >= 3 for c in components[1:]):
                        return None

                # Reject headings whose title begins with a law-code abbreviation
                # (e.g. "SGB III", "SGB III, regular hours...", "BGB") — these
                # are statutory cross-references embedded in body text, not
                # standalone clause titles.  A law-code prefix is detected as:
                #   - 2-6 uppercase letters (the abbreviation)
                #   - optionally followed by a Roman numeral (e.g. "III")
                #   - immediately followed by:
                #       end of string  → bare code "SGB III"
                #       punctuation    → "SGB III, ..."  (sentence continuation)
                #       lowercase word → "SGB III und ..." (conjunction)
                if section_title and re.match(
                    r"^[A-Z]{2,6}(?:\s+" + _ROMAN + r")?(?:\s*[,;:(]|\s+[a-z]|\s*$)",
                    section_title,
                ):
                    return None

                return section_token, section_title

            # First non-noise, non-heading line means no heading in this paragraph
            return None

        return None

    def _infer_parent_section(self, section_token: str) -> str:
        """Derive the parent section token from a numeric heading token.

        For dotted numeric headings (e.g. ``"3.2.1"``) the parent is the
        token with the last component removed (``"3.2"``). For a single-level
        token (``"3"``) or a non-numeric token (``"Section 5"``), there is no
        numeric parent and an empty string is returned.

        Args:
            section_token: The heading token for the current chunk.

        Returns:
            The parent heading token as a string, or an empty string when the
            section has no numeric parent.

        Example:
            >>> chunker = LegalSemanticChunker()
            >>> chunker._infer_parent_section("3.2.1")
            '3.2'
            >>> chunker._infer_parent_section("3")
            ''
            >>> chunker._infer_parent_section("Section 5")
            ''
        """
        numeric_match = _NUMERIC_HEADING_RE.match(section_token)

        if not numeric_match:
            return ""

        parts = numeric_match.group(1).split(".")

        if len(parts) <= 1:
            return ""

        return ".".join(parts[:-1])

    def _build_chunk(
        self,
        *,
        document_name: str,
        pages: list[int],
        section: str,
        section_title: str,
        parent_section: str,
        paragraphs: list[str],
    ) -> ChunkData:
        """Assemble a :class:`ChunkData` dict from its constituent parts.

        The ``chunk_id`` is constructed as
        ``<sanitised_document_name>_<first_page>_<section>`` to be both human
        readable and deterministic for the same document content.

        Args:
            document_name: Filename of the source PDF.
            pages: Sorted list of page numbers that contributed to this chunk.
            section: Heading token for this chunk.
            section_title: Descriptive title from the heading line.
            parent_section: Heading token of the nearest ancestor section.
            paragraphs: Ordered paragraph strings belonging to this chunk.

        Returns:
            A fully populated :class:`ChunkData` dict.
        """
        first_page = pages[0] if pages else 0

        # Sanitise document name for use in the ID (replace dots and spaces)
        doc_slug = re.sub(r"[\s\.]+", "_", document_name)
        section_slug = re.sub(r"\s+", "_", section)
        chunk_id = f"{doc_slug}_{first_page}_{section_slug}"

        return ChunkData(
            chunk_id=chunk_id,
            document_name=document_name,
            pages=sorted(set(pages)),
            section=section,
            section_title=section_title,
            parent_section=parent_section,
            chunk_text="\n\n".join(paragraphs),
        )
