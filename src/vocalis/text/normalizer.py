"""Converts raw markdown text into plain, speakable text for Document
Reader's TTS engines.

Strips markdown syntax that would otherwise be read aloud literally (a
TTS engine has no idea "**bold**" means emphasis — it would either speak
the asterisks or mangle the words) while keeping the underlying words,
and collapses excess whitespace so chunker.py sees clean paragraphs.

Order matters: code fences are removed before anything else looks at
their contents (code should never be read aloud), inline code is
protected before emphasis-stripping (so `__init__` is never mistaken for
bold), and links/images are resolved to their label before heading/
emphasis markers are stripped generally (a link's own brackets would
otherwise look like plain text containing stray brackets).

Star emphasis (*, **) uses only whitespace-adjacency guards: the char
after opening and before closing must not be whitespace. This allows
`2 * 3` to pass through unchanged (spaces flanking the * prevent matching).

Underscore emphasis (_, __) uses both whitespace-adjacency AND non-
intraword guards: underscores inside identifiers like `text_input` must
never count as emphasis, only asterisks can. This reflects standard
Markdown semantics.
"""
import re

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
# Star patterns: whitespace-adjacency guard only (no word-char guards)
_BOLD_ITALIC_STAR_RE = re.compile(r"\*\*\*(?!\s)(.+?)(?<!\s)\*\*\*")
_BOLD_STAR_RE = re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*")
_ITALIC_STAR_RE = re.compile(r"\*(?!\s)(.+?)(?<!\s)\*")
# Underscore patterns: whitespace-adjacency + non-intraword guards
_BOLD_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)___(?!\s)(.+?)(?<!\s)___(?!\w)")
_BOLD_UNDERSCORE_RE = re.compile(r"(?<!\w)__(?!\s)(.+?)(?<!\s)__(?!\w)")
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)")
_INLINE_WHITESPACE_RE = re.compile(r"[ \t]+")
_EXCESS_BLANK_LINES_RE = re.compile(r"\n{3,}")
_URL_RE = re.compile(r"\bhttps?://[^\s,;:!?)]+|\bwww\.[^\s,;:!?)]+", re.IGNORECASE)
_CURRENCY_RE = re.compile(r"\$(\d+(?:\.\d+)?)")
_PERCENT_RE = re.compile(r"(\d)%")


def _speak_url(match: "re.Match[str]") -> str:
    """Turn one matched URL into a short, spoken-friendly phrase."""
    url = re.sub(r"^https?://", "", match.group(0), flags=re.IGNORECASE)
    url = url.replace(".", " dot ").replace("/", " slash ")
    url = re.sub(r"\s+", " ", url).strip()
    return re.sub(r"\s*slash$", "", url)


def normalize_text(text: str) -> str:
    """Make raw text easier for a TTS engine to pronounce.

    Deliberately basic, and independent of normalize_markdown (Quick
    Reader's clipboard text is never markdown): speaks out URLs
    ("https://example.com/docs" -> "example dot com slash docs"), dollar
    amounts ("$42" -> "42 dollars") and percentages ("50%" -> "50
    percent"), and expands the two symbols a voice engine otherwise
    either misreads or skips (& and @). Plain numbers are left alone --
    the voice engines already read digit sequences correctly, so
    over-normalizing them would do more harm than good.

    Args:
        text: Raw text, already markdown-free (or never markdown).

    Returns:
        The same text with the substitutions above applied. Never raises.
    """
    result = _URL_RE.sub(_speak_url, text)
    result = _CURRENCY_RE.sub(r"\1 dollars", result)
    result = _PERCENT_RE.sub(r"\1 percent", result)
    result = result.replace("&", " and ").replace("@", " at ")
    return re.sub(r"[ \t]+", " ", result).strip()


def normalize_markdown(text: str) -> str:
    """Convert markdown text into plain, speakable text.

    Args:
        text: Raw markdown source (or plain text, which passes through
            unchanged since it contains none of the syntax below).

    Returns:
        Plain text with markdown syntax removed, links/images replaced
        by their label text, and excess whitespace collapsed. Never
        raises.
    """
    # NUL is never speakable and is used below as the inline-code placeholder
    # delimiter, so strip it up front to rule out any collision.
    result = _CODE_FENCE_RE.sub(" ", text.replace("\x00", ""))
    result = _IMAGE_RE.sub(r"\1", result)
    result = _LINK_RE.sub(r"\1", result)
    result = _HEADING_RE.sub("", result)

    # Protect inline code: extract and store, replace with placeholders,
    # restore literally at the end (so emphasis-stripping never touches it).
    inline_codes: list[str] = []

    def store_inline_code(match: re.Match[str]) -> str:
        inline_codes.append(match.group(1))
        return f"\x00CODE{len(inline_codes) - 1}\x00"

    result = _INLINE_CODE_RE.sub(store_inline_code, result)

    # Now strip emphasis (safe: inline code is protected by placeholders).
    result = _BOLD_ITALIC_STAR_RE.sub(r"\1", result)
    result = _BOLD_ITALIC_UNDERSCORE_RE.sub(r"\1", result)
    result = _BOLD_STAR_RE.sub(r"\1", result)
    result = _BOLD_UNDERSCORE_RE.sub(r"\1", result)
    result = _ITALIC_STAR_RE.sub(r"\1", result)
    result = _ITALIC_UNDERSCORE_RE.sub(r"\1", result)

    # Collapse whitespace.
    result = _INLINE_WHITESPACE_RE.sub(" ", result)
    result = _EXCESS_BLANK_LINES_RE.sub("\n\n", result)

    # Restore inline code: replace placeholders with original, unprocessed content.
    for i, code_text in enumerate(inline_codes):
        result = result.replace(f"\x00CODE{i}\x00", code_text)

    return normalize_text(result)
