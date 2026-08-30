from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationalState:
    predicate: str
    subject: str
    obj: str


_TRANSLATION = {
    "en": {"joseph": "joseph", "likes": "likes", "programming": "programming"},
    "sw": {"joseph": "joseph", "anapenda": "likes", "programu": "programming"},
    "ja": {"ジョセフ": "joseph", "好き": "likes", "プログラミング": "programming"},
    "zh": {"约瑟夫": "joseph", "喜欢": "likes", "编程": "programming"},
}


def normalize(language: str, subject: str, predicate: str, obj: str) -> RelationalState:
    lex = _TRANSLATION[language]
    return RelationalState(
        predicate=lex.get(predicate, predicate),
        subject=lex.get(subject, subject),
        obj=lex.get(obj, obj),
    )

