"""MONR inference for query suggestion.
This implementation aims to reproduce, as closely as practical from the
public method description, Multi-objective Neural Retrieval for Query
AutoComplete [Patki et al., 2024].
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import List, Tuple

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from cadeqs.infer import (
    build_ivfpq_index,
    load_index,
    load_test_data,
    save_index,
)
from cadeqs.util import InferenceBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


class CharacterEncoder:
    """Character-level encoder inspired by MONR."""

    def __init__(self, vector_size: int = 50):
        self.vector_size = vector_size
        # MONR appendix describes a fixed vocab:
        # 26 letters + 10 numbers + 5 symbols.
        vocab = "abcdefghijklmnopqrstuvwxyz0123456789 -_&'"
        if vector_size <= len(vocab):
            raise ValueError(
                f"char_dim={vector_size} is too small for MONR-style vocab "
                f"size {len(vocab)} (+1 unk bucket). "
                f"Use char_dim >= {len(vocab) + 1}."
            )
        self.char2idx = {c: i for i, c in enumerate(vocab)}
        # Keep one stable bucket for out-of-vocabulary characters.
        self.unk_idx = len(vocab)

    def _f(self, c: str) -> int:
        return self.char2idx.get(c, self.unk_idx)

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.vector_size, dtype=np.float32)

        text = text.strip().lower()
        p = 0
        d = 0
        for c in text:
            pos = (p + self._f(c)) % self.vector_size
            vec[pos] = math.exp(-d)
            d += 1
            p = pos

        norm_sq = float(np.dot(vec, vec))
        if norm_sq > 0.0:
            vec /= norm_sq
        return vec


def _normalize_popularity(raw_pops: np.ndarray) -> np.ndarray:
    raw = np.maximum(raw_pops.astype(np.float32), 0.0)
    log_pops = np.log1p(raw)
    max_log = float(log_pops.max()) if len(log_pops) else 1.0
    if max_log <= 0.0:
        max_log = 1.0
    return (log_pops / max_log).astype(np.float32).reshape(-1, 1)


def load_corpus(corpus_path: str) -> tuple[List[str], np.ndarray]:
    """Load corpus from JSONL with required keys: query, frequency."""
    with open(corpus_path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    queries = [row["query"] for row in rows]
    pops = [float(row["frequency"]) for row in rows]

    if not queries:
        raise ValueError(f"Corpus is empty: {corpus_path}")
    return queries, np.array(pops, dtype=np.float32)


def build_feature_matrix(
    queries: List[str],
    raw_popularity: np.ndarray,
    model: SentenceTransformer,
    char_encoder: CharacterEncoder,
    *,
    semantic_weight: float,
    char_weight: float,
    pop_weight: float,
    l2_normalize_concat: bool,
    batch_size: int = 8192,
) -> np.ndarray:
    logger.info("Encoding semantic vectors...")
    sem_embs = model.encode(
        queries,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    logger.info("Encoding character vectors...")
    char_embs = np.array(
        [char_encoder.encode(q) for q in tqdm(queries, desc="Char encoding")],
        dtype=np.float32,
    )
    pop_embs = _normalize_popularity(raw_popularity)

    features = np.concatenate(
        [
            semantic_weight * sem_embs,
            char_weight * char_embs,
            pop_weight * pop_embs,
        ],
        axis=1,
    ).astype(np.float32)
    if l2_normalize_concat:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features /= np.clip(norms, 1e-12, None)
    return features


def _split_query(query: str, sep: str) -> Tuple[str, str]:
    parts = query.split(sep)
    prefix = parts[0].strip() if parts else ""
    context = parts[1].strip() if len(parts) >= 2 else ""
    return prefix, context


def suggest_per_query(
    query: str,
    model: SentenceTransformer,
    char_encoder: CharacterEncoder,
    index: faiss.Index,
    id2query: dict[int, str],
    *,
    top_k: int,
    context_sep: str,
    semantic_dim: int,
    semantic_weight: float,
    char_weight: float,
    pop_weight: float,
    l2_normalize_concat: bool,
) -> List[str]:
    """Build MONR query vector and retrieve top-k candidates."""
    prefix, context = _split_query(query, context_sep)

    if context:
        sem = model.encode(
            [context],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
    else:
        sem = np.zeros(semantic_dim, dtype=np.float32)
    char = char_encoder.encode(prefix)
    pop = np.array([1.0], dtype=np.float32)

    query_vec = np.concatenate(
        [
            semantic_weight * sem,
            char_weight * char,
            pop_weight * pop,
        ]
    ).astype(np.float32)
    if l2_normalize_concat:
        query_vec /= max(float(np.linalg.norm(query_vec)), 1e-12)

    _, hits = index.search(np.array([query_vec], dtype=np.float32), top_k)
    return [id2query.get(int(idx), "") for idx in hits[0] if int(idx) >= 0]


def run(args: argparse.Namespace) -> None:
    logger.info("Loading MONR encoder: %s", args.model_path)
    model = SentenceTransformer(
        args.model_path,
        device=args.device,
        trust_remote_code=True,
    )
    semantic_dim = model.get_sentence_embedding_dimension()
    char_encoder = CharacterEncoder(vector_size=args.char_dim)

    has_index = (
        args.index_path and os.path.exists(args.index_path) and not args.overwrite_index
    )

    if has_index:
        index, id2query = load_index(args.index_path)
        index.nprobe = args.nprobe
    else:
        logger.info("Loading corpus: %s", args.corpus_path)
        queries, pops = load_corpus(args.corpus_path)
        logger.info("Corpus size: %d", len(queries))

        embs = build_feature_matrix(
            queries,
            pops,
            model,
            char_encoder,
            semantic_weight=args.semantic_weight,
            char_weight=args.char_weight,
            pop_weight=args.pop_weight,
            l2_normalize_concat=args.l2_normalize_concat,
            batch_size=args.batch_size,
        )
        index = build_ivfpq_index(
            embs,
            nlist=args.nlist,
            m=args.m,
            nbits=args.nbits,
            nprobe=args.nprobe,
            train_sample_size=args.train_sample_size,
        )
        id2query = {i: q for i, q in enumerate(queries)}
        if args.index_path:
            os.makedirs(os.path.dirname(args.index_path) or ".", exist_ok=True)
            save_index(index, id2query, args.index_path)

    logger.info("Loading test data: %s", args.test_file)
    test_rows = load_test_data(args.test_file, context_sep=args.context_sep)
    logger.info("Test examples: %d", len(test_rows))

    logger.info("Searching top-%d candidates (per-query)...", args.top_k)
    results: list[dict] = []
    with InferenceBenchmark(
        project_name="monr_inference",
        track_energy=args.track_energy,
    ) as bench:
        for row in tqdm(test_rows, desc="Inference"):
            pred = bench.timed_call(
                suggest_per_query,
                row["query"],
                model,
                char_encoder,
                index,
                id2query,
                top_k=args.top_k,
                context_sep=args.context_sep,
                semantic_dim=semantic_dim,
                semantic_weight=args.semantic_weight,
                char_weight=args.char_weight,
                pop_weight=args.pop_weight,
                l2_normalize_concat=args.l2_normalize_concat,
            )
            results.append(
                {
                    "query": row["query"],
                    "gold_candidate": row["gold_candidate"],
                    "pred_candidates": pred,
                }
            )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    stats_path = os.path.splitext(args.output_file)[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(bench.summary(), f, ensure_ascii=False, indent=2)
    logger.info("Stats written to %s", stats_path)
    logger.info(
        "Results written to %s (%d lines)",
        args.output_file,
        len(results),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MONR inference for query suggestion",
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to MONR sentence-transformer checkpoint",
    )
    p.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Candidate corpus JSONL with query/frequency",
    )
    p.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Test JSONL file with context/target",
    )
    p.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSONL path",
    )
    p.add_argument(
        "--index_path",
        type=str,
        default=None,
        help="Path to save/load FAISS index",
    )
    p.add_argument("--overwrite_index", action="store_true")
    p.add_argument("--track_energy", action="store_true")

    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--context_sep", type=str, default="[SEP]")

    p.add_argument("--semantic_weight", type=float, default=1.0)
    p.add_argument("--char_weight", type=float, default=1.0)
    p.add_argument("--pop_weight", type=float, default=1.0)
    p.add_argument(
        "--char_dim",
        type=int,
        default=63,
        help=(
            "Character feature dimension. Default 63 assumes semantic_dim=768 and "
            "m=64 (D=768+63+1=832). If semantic_dim or m differ, adjust char_dim "
            "so D=(semantic_dim+char_dim+1) is divisible by m for IVFPQ."
        ),
    )
    p.add_argument(
        "--l2_normalize_concat",
        action="store_true",
        help=(
            "L2-normalize concatenated [semantic, char, popularity] vectors "
            "for both corpus and query. Default: disabled."
        ),
    )

    p.add_argument("--nlist", type=int, default=50000)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--nbits", type=int, default=8)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--train_sample_size", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
