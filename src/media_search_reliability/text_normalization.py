from __future__ import annotations

import re
import unicodedata
from typing import Iterable

_GENRE_ALIASES = {
    "sci fi": "sci-fi",
    "science fiction": "sci-fi",
    "scifi": "sci-fi",
    "film noir": "film-noir",
    "childrens": "children's",
    "children": "children's",
}

_TAG_ALIASES = {
    "robots": "robot",
    "robotic": "robot",
    "robotics": "robot",
    "humorous": "funny",
    "comedic": "funny",
    "comedy": "funny",
    "family friendly": "family",
    "family-friendly": "family",
    "fast paced": "fast-paced",
    "fast-paced": "fast-paced",
    "thought provoking": "psychological",
    "thought-provoking": "psychological",
    "suspense": "mystery",
}


def normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[!?.,;:'\"()\[\]{}]", " ", text)
    text = re.sub(r"[-_/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_alias(text: str, aliases: dict[str, str]) -> str:
    return aliases.get(text, text)


def canonical_genre(value: object) -> str:
    return _apply_alias(normalize_text(value), _GENRE_ALIASES)


def canonical_tag(value: object) -> str:
    text = normalize_text(value)
    text = _apply_alias(text, _TAG_ALIASES)
    if text.endswith("s") and len(text) > 4 and text[:-1] in {"robot", "alien", "detective"}:
        text = text[:-1]
    return text



def canonical_genre_phrase(value: object) -> str:
    text = normalize_text(value)
    for source, target in sorted(_GENRE_ALIASES.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{re.escape(source)}\b", target.replace("-", " "), text)
    return re.sub(r"\s+", " ", text).strip()

def canonical_phrase(value: object) -> str:
    text = normalize_text(value)
    # Replace multi-word aliases before token-level matching.
    for source, target in sorted({**_GENRE_ALIASES, **_TAG_ALIASES}.items(), key=lambda x: -len(x[0])):
        text = re.sub(rf"\b{re.escape(source)}\b", target.replace("-", " "), text)
    return re.sub(r"\s+", " ", text).strip()


def split_pipe(value: object, *, kind: str = "tag") -> list[str]:
    fn = canonical_genre if kind == "genre" else canonical_tag
    parts = []
    for raw in str(value).split("|"):
        item = fn(raw)
        if item and item not in {"nan", "no genres listed"}:
            parts.append(item)
    return parts


def canonical_join(values: Iterable[object], *, kind: str = "tag") -> str:
    fn = canonical_genre if kind == "genre" else canonical_tag
    out = []
    for value in values:
        item = fn(value)
        if item and item not in out:
            out.append(item)
    return "|".join(out)
