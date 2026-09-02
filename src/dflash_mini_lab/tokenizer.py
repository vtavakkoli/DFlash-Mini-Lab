from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

TOKEN_RE = re.compile(r"<eos>|[A-Za-z0-9]+|[^\w\s]")


@dataclass(frozen=True)
class WordTokenizer:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(stoi={str(k): int(v) for k, v in data["stoi"].items()}, itos=list(data["itos"]))

    @property
    def bos_id(self) -> int:
        return self.stoi["<bos>"]

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in TOKEN_RE.findall(text.lower())]
        return ([self.bos_id] + ids) if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        toks = [self.itos[i] for i in ids if 0 <= i < len(self.itos) and self.itos[i] not in {"<pad>", "<bos>"}]
        out = " ".join(toks)
        for punct in [".", ",", "!", "?", ":", ";"]:
            out = out.replace(f" {punct}", punct)
        return out
