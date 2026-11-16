"""Streaming normalization, quality filtering, and exact deduplication for Vietnamese Wikipedia."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache

DEFAULT_STRIP_SECTIONS: tuple[str, ...] = (
    "Xem thêm",
    "Tham khảo",
    "Chú thích",
    "Liên kết ngoài",
    "Đọc thêm",
    "Tài liệu tham khảo",
    "Nguồn",
)

_EXOTIC_SPACES_RE = re.compile(r"[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_BLOCK_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_REF_SELF_RE = re.compile(r"<ref\b[^>]*/>", re.IGNORECASE)
_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>\n]*>")
_WIKI_FILE_RE = re.compile(r"\[\[(?:File|Tập tin|Hình|Image):[^\]\n]*(?:\[\[[^\]\n]*\]\][^\]\n]*)*\]\]", re.IGNORECASE)
_WIKI_TABLE_LINE_RE = re.compile(r"(?m)^[ \t]*(?:\{\||\|\}|\|-|!|\|[ \t]).*$")
_INLINE_TEMPLATE_RE = re.compile(r"\{\{[^{}\n]{0,200}\}\}")
_HSPACE_RE = re.compile(r"[ \t]+")
_LINE_EDGE_SPACE_RE = re.compile(r"[ \t]*\n[ \t]*")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

_ALLOWED_PUNCT = frozenset(".,;:!?'\"()[]{}-–—/\\%+=*&@#~_…“”‘’«»°•·|<>`")
_WHITESPACE_CHARS = frozenset(" \n\t")


@lru_cache(maxsize=64)
def _compile_section_strip_regex(sections: tuple[str, ...]) -> re.Pattern[str] | None:
    """Compile a regex matching Wikipedia trailing section headings at line start."""
    cleaned = [unicodedata.normalize("NFC", s.strip()) for s in sections if s and s.strip()]
    if not cleaned:
        return None
    escaped = "|".join(re.escape(s) for s in cleaned)
    return re.compile(
        rf"(?:^|\n)[ \t]*(?:={{2,6}}[ \t]*|#{{1,6}}[ \t]*)?(?:{escaped})(?:[ \t]*={{2,6}})?[ \t]*:?[ \t]*(?:\n|$)",
        re.IGNORECASE,
    )


def _strip_invisible_controls(text: str) -> str:
    """Remove invisible control and format chars (Cc/Cf) while keeping newline and tab."""
    out: list[str] = []
    for ch in text:
        if ch == "\n" or ch == "\t":
            out.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat in ("Cc", "Cf"):
            continue
        out.append(ch)
    return "".join(out)


def normalize_text(
    text: str,
    strip_sections: Sequence[str] | None = DEFAULT_STRIP_SECTIONS,
) -> str:
    """Normalize Vietnamese Wikipedia text to NFC, clean markup remnants, and preserve paragraphs.

    Args:
        text: Raw document string.
        strip_sections: Section headings whose appearance at line start truncates the remainder
            of the document (e.g. "Xem thêm", "Tham khảo"). Pass empty sequence or None to disable.

    Returns:
        Cleaned, NFC-normalized text with paragraph breaks (`\\n\\n`) preserved.
    """
    if not text:
        return ""

    # Standardize line endings and Unicode canonical composition (NFC) first.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)

    # Remove invisible control/format chars (e.g. ZWSP, BOM, soft hyphen) except \n and \t.
    text = _strip_invisible_controls(text)
    text = _EXOTIC_SPACES_RE.sub(" ", text)

    # Clean common HTML/Wiki markup leftovers while keeping ordinary prose and <|endoftext|>.
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _REF_BLOCK_RE.sub(" ", text)
    text = _REF_SELF_RE.sub(" ", text)
    text = _BR_TAG_RE.sub("\n", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WIKI_FILE_RE.sub(" ", text)
    text = _WIKI_TABLE_LINE_RE.sub("", text)
    text = _INLINE_TEMPLATE_RE.sub(" ", text)

    # Truncate trailing Wikipedia reference/navigation sections if configured.
    if strip_sections:
        sec_re = _compile_section_strip_regex(tuple(strip_sections))
        if sec_re is not None:
            match = sec_re.search(text)
            if match is not None:
                text = text[: match.start()]

    # Collapse redundant horizontal whitespace while preserving paragraph structure.
    text = _HSPACE_RE.sub(" ", text)
    text = _LINE_EDGE_SPACE_RE.sub("\n", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


@lru_cache(maxsize=8192)
def _char_class(ch: str) -> int:
    """Classify a single char: 2 = Latin/Vietnamese letter, 1 = other valid char, 0 = invalid."""
    if ch in _WHITESPACE_CHARS or ch in _ALLOWED_PUNCT:
        return 1
    cat = unicodedata.category(ch)
    if cat == "Nd":
        return 1
    if cat.startswith("L"):
        # Vietnamese NFC letters all belong to the Latin script block/names.
        if "LATIN" in unicodedata.name(ch, ""):
            return 2
        return 0
    return 0


def text_quality_metrics(text: str) -> tuple[float, float]:
    """Compute (valid_char_ratio, latin_letter_ratio) over all characters in text."""
    total = len(text)
    if total == 0:
        return 0.0, 0.0

    valid_count = 0
    letter_count = 0
    for ch in text:
        cls = _char_class(ch)
        if cls >= 1:
            valid_count += 1
            if cls == 2:
                letter_count += 1
    return valid_count / total, letter_count / total


def filter_document(
    text: str,
    min_chars: int = 250,
    min_valid_char_ratio: float = 0.85,
    min_letter_ratio: float = 0.45,
) -> bool:
    """Return True if normalized text meets length and Vietnamese/Latin character ratio thresholds."""
    if len(text) < min_chars:
        return False
    valid_ratio, letter_ratio = text_quality_metrics(text)
    return valid_ratio >= min_valid_char_ratio and letter_ratio >= min_letter_ratio


def filter_documents(
    docs: Iterable[str],
    min_chars: int = 250,
    min_valid_char_ratio: float = 0.85,
    min_letter_ratio: float = 0.45,
) -> Iterator[str]:
    """Yield documents from `docs` that pass quality filtering (streaming, zero-OOM)."""
    for doc in docs:
        if filter_document(
            doc,
            min_chars=min_chars,
            min_valid_char_ratio=min_valid_char_ratio,
            min_letter_ratio=min_letter_ratio,
        ):
            yield doc


def dedup_exact(docs: Iterable[str]) -> Iterator[str]:
    """Yield unique documents using exact SHA-256 hashing over UTF-8 bytes."""
    seen_hashes: set[bytes] = set()
    for doc in docs:
        digest = hashlib.sha256(doc.encode("utf-8")).digest()
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            yield doc


def clean_and_dedup_stream(
    docs: Iterable[str],
    min_chars: int = 250,
    min_valid_char_ratio: float = 0.85,
    min_letter_ratio: float = 0.45,
    strip_sections: Sequence[str] | None = DEFAULT_STRIP_SECTIONS,
) -> Iterator[str]:
    """Run full streaming pipeline: normalize -> filter -> exact dedup."""
    normalized = (normalize_text(doc, strip_sections=strip_sections) for doc in docs)
    filtered = filter_documents(
        normalized,
        min_chars=min_chars,
        min_valid_char_ratio=min_valid_char_ratio,
        min_letter_ratio=min_letter_ratio,
    )
    yield from dedup_exact(filtered)
