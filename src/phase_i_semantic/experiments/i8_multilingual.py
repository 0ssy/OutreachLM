from __future__ import annotations

from typing import Any

from src.phase_i_semantic.semantics.multilingual_relay import normalize


def run() -> dict[str, Any]:
    en = normalize("en", "joseph", "likes", "programming")
    sw = normalize("sw", "joseph", "anapenda", "programu")
    ja = normalize("ja", "ジョセフ", "好き", "プログラミング")
    zh = normalize("zh", "约瑟夫", "喜欢", "编程")

    en_sw_match = 1.0 if en == sw else 0.0
    ja_zh_match = 1.0 if ja == zh else 0.0
    status = "PASS" if en_sw_match >= 1.0 and ja_zh_match >= 1.0 else "FAIL"
    return {
        "english_to_swahili_state_match_rate": en_sw_match,
        "japanese_to_mandarin_state_match_rate": ja_zh_match,
        "cross_lingual_relation_invariance_status": status,
    }

