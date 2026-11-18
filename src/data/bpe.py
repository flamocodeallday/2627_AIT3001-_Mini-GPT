"""Self-implemented Byte-level Regex-BPE tokenizer optimized for Vietnamese corpora."""

from __future__ import annotations

import base64
import heapq
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import regex as re

from src.data.tokenizer_base import EOS_TOKEN, BaseTokenizer

DEFAULT_VI_BPE_PATTERN = (
    r"\r\n|\n+|"
    r" ?[\p{L}\p{M}]+|"
    r" ?\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+|"
    r"[ \t]+(?!\S)|[ \t]+|\s+"
)


def _count_adjacent_pairs(seq: Sequence[int]) -> Counter[tuple[int, int]]:
    """Count adjacent token pairs within a single token-ID sequence."""
    counts: Counter[tuple[int, int]] = Counter()
    for i in range(len(seq) - 1):
        counts[(seq[i], seq[i + 1])] += 1
    return counts


def _apply_merge_to_seq(seq: list[int], pair: tuple[int, int], merged_id: int) -> list[int]:
    """Replace all non-overlapping occurrences of `pair` in `seq` with `merged_id`."""
    a, b = pair
    n = len(seq)
    if n < 2:
        return seq

    out: list[int] = []
    i = 0
    while i < n:
        if i + 1 < n and seq[i] == a and seq[i + 1] == b:
            out.append(merged_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


class CustomBPE(BaseTokenizer):
    """Byte-level Regex-BPE tokenizer with frequency-weighted training and memoized encoding.

    Token ID layout:
      - 0..255: raw single UTF-8 bytes
      - 256: `<|endoftext|>` (`eos_id`)
      - 257..(vocab_size - 1): learned BPE merge tokens
    """

    def __init__(self, pattern: str = DEFAULT_VI_BPE_PATTERN) -> None:
        self.pattern: str = pattern
        self._chunk_re = re.compile(pattern)

        self._eos_id: int = 256
        self.special_tokens: dict[str, int] = {EOS_TOKEN: self._eos_id}
        self._id_to_special: dict[int, str] = {self._eos_id: EOS_TOKEN}
        self._special_re = re.compile(f"({re.escape(EOS_TOKEN)})")

        self._vocab_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self._merges: list[tuple[int, int, int]] = []
        self._merge_ranks: dict[tuple[int, int], tuple[int, int]] = {}
        self._vocab_size: int = 257
        self._encode_cache: dict[bytes, tuple[int, ...]] = {}
        self._max_cache_entries: int = 250_000

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def merges(self) -> list[tuple[int, int, int]]:
        """Return learned merges as `(left_id, right_id, merged_id)` triples."""
        return list(self._merges)

    def _reset_learned_state(self) -> None:
        self._vocab_bytes = {i: bytes([i]) for i in range(256)}
        self._merges = []
        self._merge_ranks = {}
        self._vocab_size = 257
        self._encode_cache.clear()

    def _collect_chunk_frequencies(self, texts: Iterable[str]) -> Counter[bytes]:
        """Pre-tokenize streaming texts and count unique UTF-8 chunk frequencies."""
        chunk_freqs: Counter[bytes] = Counter()
        finditer = self._chunk_re.finditer
        for text in texts:
            if not text:
                continue
            for match in finditer(text):
                chunk_freqs[match.group(0).encode("utf-8")] += 1
        return chunk_freqs

    def train(self, texts: Iterable[str], vocab_size: int = 8192) -> None:
        """Train BPE merges on streaming `texts` up to `vocab_size` (must be in [257, 65535])."""
        if vocab_size < 257 or vocab_size > 65535:
            raise ValueError(f"vocab_size must be between 257 and 65535, got {vocab_size}")

        self._reset_learned_state()
        target_merges = vocab_size - 257
        if target_merges == 0:
            return

        chunk_freqs = self._collect_chunk_frequencies(texts)
        if not chunk_freqs:
            return

        # Keep only multi-byte chunks; 1-byte chunks can never form an adjacent pair.
        seqs: list[list[int]] = []
        freqs: list[int] = []
        for chunk_bytes, count in chunk_freqs.items():
            if len(chunk_bytes) >= 2:
                seqs.append(list(chunk_bytes))
                freqs.append(count)

        if not seqs:
            return

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_chunks: dict[tuple[int, int], set[int]] = defaultdict(set)
        for cid, (seq, weight) in enumerate(zip(seqs, freqs)):
            local_pairs = _count_adjacent_pairs(seq)
            for pair, occ in local_pairs.items():
                pair_counts[pair] += occ * weight
                pair_to_chunks[pair].add(cid)

        # Max-heap via negative count; deterministic tie-breaking by pair tuple.
        heap: list[tuple[int, tuple[int, int]]] = [
            (-cnt, pair) for pair, cnt in pair_counts.items() if cnt >= 2
        ]
        heapq.heapify(heap)

        next_id = 257
        for rank in range(target_merges):
            best_pair: tuple[int, int] | None = None
            while heap:
                neg_cnt, candidate = heapq.heappop(heap)
                cur_cnt = pair_counts.get(candidate, 0)
                if cur_cnt >= 2 and -neg_cnt == cur_cnt:
                    best_pair = candidate
                    break

            if best_pair is None:
                break

            merged_id = next_id
            next_id += 1

            a, b = best_pair
            self._merges.append((a, b, merged_id))
            self._merge_ranks[best_pair] = (rank, merged_id)
            self._vocab_bytes[merged_id] = self._vocab_bytes[a] + self._vocab_bytes[b]

            affected_ids = list(pair_to_chunks.pop(best_pair, ()))
            for cid in affected_ids:
                old_seq = seqs[cid]
                if len(old_seq) < 2:
                    continue
                new_seq = _apply_merge_to_seq(old_seq, best_pair, merged_id)
                if new_seq == old_seq:
                    continue

                weight = freqs[cid]
                old_local = _count_adjacent_pairs(old_seq)
                new_local = _count_adjacent_pairs(new_seq)
                seqs[cid] = new_seq

                for pair in old_local.keys() | new_local.keys():
                    old_occ = old_local.get(pair, 0)
                    new_occ = new_local.get(pair, 0)
                    if old_occ == new_occ:
                        continue

                    delta = (new_occ - old_occ) * weight
                    updated = pair_counts.get(pair, 0) + delta
                    if updated <= 0:
                        pair_counts.pop(pair, None)
                        pair_to_chunks.pop(pair, None)
                    else:
                        pair_counts[pair] = updated
                        if old_occ == 0 and new_occ > 0:
                            pair_to_chunks[pair].add(cid)
                        elif old_occ > 0 and new_occ == 0:
                            chunk_set = pair_to_chunks.get(pair)
                            if chunk_set is not None:
                                chunk_set.discard(cid)
                        if updated >= 2:
                            heapq.heappush(heap, (-updated, pair))

            pair_counts.pop(best_pair, None)

        self._vocab_size = 257 + len(self._merges)

    def _encode_chunk_bytes(self, chunk_bytes: bytes) -> tuple[int, ...]:
        """Apply learned BPE merges to one pre-tokenized UTF-8 chunk with memoization."""
        cached = self._encode_cache.get(chunk_bytes)
        if cached is not None:
            return cached

        if len(chunk_bytes) == 1:
            result = (chunk_bytes[0],)
        else:
            ids = list(chunk_bytes)
            merge_ranks = self._merge_ranks
            while len(ids) >= 2:
                best_pair: tuple[int, int] | None = None
                best_rank = -1
                best_merged = -1
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i + 1])
                    info = merge_ranks.get(pair)
                    if info is not None:
                        rank, merged_id = info
                        if best_rank == -1 or rank < best_rank:
                            best_rank = rank
                            best_merged = merged_id
                            best_pair = pair
                if best_pair is None:
                    break
                ids = _apply_merge_to_seq(ids, best_pair, best_merged)
            result = tuple(ids)

        if len(chunk_bytes) <= 64 and len(self._encode_cache) < self._max_cache_entries:
            self._encode_cache[chunk_bytes] = result
        return result

    def _encode_ordinary(self, text: str) -> list[int]:
        out: list[int] = []
        for match in self._chunk_re.finditer(text):
            chunk_bytes = match.group(0).encode("utf-8")
            out.extend(self._encode_chunk_bytes(chunk_bytes))
        return out

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        """Encode text into BPE token IDs; literal special text is byte-encoded unless allowed."""
        if not text:
            return []
        if not allow_special:
            return self._encode_ordinary(text)

        out: list[int] = []
        for segment in self._special_re.split(text):
            if not segment:
                continue
            special_id = self.special_tokens.get(segment)
            if special_id is not None:
                out.append(special_id)
            else:
                out.extend(self._encode_ordinary(segment))
        return out

    def decode(self, ids: Sequence[int]) -> str:
        """Decode BPE token IDs back to UTF-8 text, handling partial byte sequences safely."""
        out_parts: list[str] = []
        byte_buf = bytearray()

        for tid in ids:
            tid_int = int(tid)
            if tid_int in self._id_to_special:
                if byte_buf:
                    out_parts.append(byte_buf.decode("utf-8", errors="replace"))
                    byte_buf.clear()
                out_parts.append(self._id_to_special[tid_int])
            elif tid_int in self._vocab_bytes:
                byte_buf.extend(self._vocab_bytes[tid_int])
            else:
                raise ValueError(f"Token ID {tid_int} out of range [0, {self._vocab_size - 1}]")

        if byte_buf:
            out_parts.append(byte_buf.decode("utf-8", errors="replace"))
        return "".join(out_parts)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize tokenizer config, Base64 byte tokens, and merges into a single JSON dict."""
        serialized_merges = [
            {
                "id": merged_id,
                "pair": [left_id, right_id],
                "bytes_b64": base64.b64encode(self._vocab_bytes[merged_id]).decode("ascii"),
            }
            for left_id, right_id, merged_id in self._merges
        ]
        return {
            "tokenizer_type": "custom_regex_bpe",
            "vocab_size": self._vocab_size,
            "eos_token": self.eos_token,
            "eos_id": self._eos_id,
            "pattern": self.pattern,
            "special_tokens": dict(self.special_tokens),
            "merges": serialized_merges,
        }

    def _load_json_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("tokenizer_type") != "custom_regex_bpe":
            raise ValueError(
                f"Expected tokenizer_type='custom_regex_bpe', got {payload.get('tokenizer_type')!r}"
            )

        pattern = str(payload["pattern"])
        eos_id = int(payload["eos_id"])
        eos_token = str(payload.get("eos_token", EOS_TOKEN))
        special_tokens = {str(k): int(v) for k, v in payload["special_tokens"].items()}
        declared_vocab_size = int(payload["vocab_size"])

        self.pattern = pattern
        self._chunk_re = re.compile(pattern)
        self.eos_token = eos_token
        self._eos_id = eos_id
        self.special_tokens = special_tokens
        self._id_to_special = {v: k for k, v in special_tokens.items()}
        self._special_re = re.compile("(" + "|".join(re.escape(k) for k in special_tokens) + ")")

        self._reset_learned_state()
        for rank, item in enumerate(payload["merges"]):
            merged_id = int(item["id"])
            left_id, right_id = int(item["pair"][0]), int(item["pair"][1])
            expected_bytes = self._vocab_bytes[left_id] + self._vocab_bytes[right_id]
            stored_bytes = base64.b64decode(item["bytes_b64"].encode("ascii"))
            if stored_bytes != expected_bytes:
                raise ValueError(
                    f"Corrupted merge entry for token ID {merged_id}: bytes mismatch"
                )
            self._merges.append((left_id, right_id, merged_id))
            self._merge_ranks[(left_id, right_id)] = (rank, merged_id)
            self._vocab_bytes[merged_id] = stored_bytes

        self._vocab_size = 257 + len(self._merges)
        if self._vocab_size != declared_vocab_size:
            raise ValueError(
                f"Declared vocab_size={declared_vocab_size} != loaded {self._vocab_size}"
            )
