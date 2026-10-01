"""Generated token ids, kept beside a scoring's `responses.jsonl` in a compact text file.

Why a separate file. A budget audit needs the exact tokens a model generated: the score at a shorter answer budget
is read from a prefix of ONE long generation, and decoding a prefix needs the ids, because re-tokenising decoded text
does not always give the original tokens back. Putting 8,192 integers in every row of `responses.jsonl` would take a
1,147-question scoring past the collector's 50 MB limit, and the collector drops an over-limit file whole, answers
included. So the ids live in `tokens.jsonl`, one row per answer, each id list packed as little-endian unsigned 32-bit
integers, zlib-compressed and base64-encoded: about 20 KB for 8,192 ids, text-safe, and readable with the standard
library alone.
"""
from __future__ import annotations

import base64
import json
import struct
import zlib
from pathlib import Path

ENCODING = "base64(zlib(little-endian uint32))"
FILE_NAME = "tokens.jsonl"


def pack(ids) -> str:
    """A list of non-negative token ids as one ASCII string."""
    ids = [int(i) for i in ids]
    if any(i < 0 or i > 0xFFFFFFFF for i in ids):
        raise ValueError("a token id is outside the unsigned 32-bit range")
    return base64.b64encode(zlib.compress(struct.pack("<%dI" % len(ids), *ids), 6)).decode("ascii")


def unpack(text: str) -> list:
    """The inverse of `pack`. Refuses a payload that is not a whole number of 32-bit integers."""
    raw = zlib.decompress(base64.b64decode(text.encode("ascii")))
    if len(raw) % 4:
        raise ValueError("%d bytes is not a whole number of 32-bit token ids" % len(raw))
    return list(struct.unpack("<%dI" % (len(raw) // 4), raw))


def write(path, rows) -> int:
    """`rows`: dicts carrying `ids` (a list of ints) and any identifying keys. Returns the number written."""
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            ids = list(row["ids"])
            handle.write(json.dumps({**{k: v for k, v in row.items() if k != "ids"}, "n": len(ids), "ids": pack(ids)},
                                    ensure_ascii=False) + "\n")
            count += 1
    return count


def read(path) -> list:
    """Every row with `ids` unpacked. Refuses a row whose stated count disagrees with its payload."""
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        ids = unpack(row["ids"])
        if len(ids) != row.get("n", len(ids)):
            raise ValueError("%s line %d: %d ids, but the row says %s" % (path, number, len(ids), row.get("n")))
        rows.append({**row, "ids": ids})
    return rows
