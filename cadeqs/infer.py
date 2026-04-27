from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
from typing import List

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from cadeqs.model import pool_cls
from cadeqs.util import InferenceBenchmark

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int = 8192,
    max_length: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode texts into L2-normalised embeddings using CLS pooling."""
    all_embs: list[np.ndarray] = []
    model.eval()
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Encoding")

    with torch.no_grad():
        for i in iterator:
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc, return_dict=True)
            emb = pool_cls(out.last_hidden_state)  # (B, H)
            emb = F.normalize(emb, p=2, dim=-1)
            all_embs.append(emb.cpu().numpy().astype(np.float32))

    return np.vstack(all_embs)


def build_ivfpq_index(
    embeddings: np.ndarray,
    nlist: int = 100,
    m: int = 8,
    nbits: int = 8,
    nprobe: int = 24,
    train_sample_size: int | None = None,
) -> faiss.Index:
    """Build a FAISS IVFPQ index from embeddings (assumed L2-normalised)."""
    N, D = embeddings.shape
    # Clamp nlist to corpus size
    nlist = min(nlist, N)
    # m must divide D
    if D % m != 0:
        raise ValueError(f"PQ sub-vector count m={m} must divide embedding dim D={D}. ")

    train_embs = embeddings
    if train_sample_size and N > train_sample_size:
        indices = random.sample(range(N), train_sample_size)
        train_embs = embeddings[indices]

    quantizer = faiss.IndexFlatL2(D)
    index = faiss.IndexIVFPQ(quantizer, D, nlist, m, nbits)
    index.cp.niter = 10
    index.cp.max_points_per_centroid = 64
    index.pq.train_type = faiss.ProductQuantizer.Train_shared
    logger.info(
        "Training IVFPQ index: D=%d, nlist=%d, m=%d, nbits=%d, N_train=%d",
        D,
        nlist,
        m,
        nbits,
        len(train_embs),
    )
    res = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.useFloat16LookupTables = True
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index, co)
    gpu_index.train(train_embs)
    gpu_index.add(embeddings)
    index = faiss.index_gpu_to_cpu(gpu_index)
    index.nprobe = nprobe
    logger.info(
        "Index built via GPU train/add then moved to CPU: ntotal=%d, nprobe=%d",
        index.ntotal,
        nprobe,
    )
    return index


def save_index(index: faiss.Index, id2query: dict, index_path: str) -> None:
    faiss.write_index(index, index_path)
    with open(index_path + ".meta", "wb") as f:
        pickle.dump(id2query, f)
    logger.info("Index saved: %s", index_path)


def load_index(index_path: str) -> tuple[faiss.Index, dict]:
    index = faiss.read_index(index_path)
    with open(index_path + ".meta", "rb") as f:
        id2query = pickle.load(f)
    logger.info("Index loaded: %s (ntotal=%d)", index_path, index.ntotal)
    return index, id2query


def suggest_per_query(
    query: str,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    index: faiss.Index,
    id2query: dict[int, str],
    device: str,
    top_k: int,
    max_length: int = 32,
) -> List[str]:
    """Encode a single query and retrieve top-k candidates via ANN search."""
    query_emb = encode_texts(
        [query],
        tokenizer,
        model,
        device,
        batch_size=1,
        max_length=max_length,
    )
    _, hits = index.search(query_emb, top_k)
    return [id2query.get(int(idx), "") for idx in hits[0]]


def load_corpus(corpus_path: str) -> List[str]:
    """
    Load candidate corpus from a JSONL file.
    """
    with open(corpus_path, encoding="utf-8") as f:
        return [
            obj["query"] for obj in (json.loads(line) for line in f) if "query" in obj
        ]


def load_test_data(
    test_path: str, context_sep: str = "[SEP]", max_ctx: int | None = None
) -> List[dict]:
    """Load test JSONL and join (optionally truncated) context into a query."""
    with open(test_path, encoding="utf-8") as f:
        objs = [json.loads(line) for line in f if line.strip()]

    rows: list[dict] = []
    for obj in objs:
        ctx = obj["context"]
        if max_ctx is not None:
            # Context is ordered new -> old, so take the first max_ctx items.
            ctx = ctx[:max_ctx]
        query = context_sep.join(ctx) if ctx else ""
        rows.append({"query": query, "gold_candidate": obj["target"]})
    return rows


def run(args: argparse.Namespace) -> None:
    device = args.device

    # ── Load models ──────────────────────────────────────────────────
    candidate_encoder_path = args.candidate_encoder_path or args.query_encoder_path

    logger.info("Loading query encoder: %s", args.query_encoder_path)
    query_tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path)
    query_model = AutoModel.from_pretrained(args.query_encoder_path).eval().to(device)

    if candidate_encoder_path == args.query_encoder_path:
        cand_tokenizer, cand_model = query_tokenizer, query_model
    else:
        logger.info("Loading candidate encoder: %s", candidate_encoder_path)
        cand_tokenizer = AutoTokenizer.from_pretrained(candidate_encoder_path)
        cand_model = AutoModel.from_pretrained(candidate_encoder_path).eval().to(device)

    # ── Build or load FAISS index ────────────────────────────────────
    index: faiss.Index
    id2query: dict[int, str]

    if args.index_path and os.path.exists(args.index_path) and not args.overwrite_index:
        index, id2query = load_index(args.index_path)
        index.nprobe = args.nprobe
    else:
        logger.info("Loading corpus: %s", args.corpus_path)
        corpus = load_corpus(args.corpus_path)
        logger.info("Corpus size: %d", len(corpus))

        logger.info("Encoding corpus with candidate encoder...")
        corpus_embs = encode_texts(
            corpus,
            cand_tokenizer,
            cand_model,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            show_progress=True,
        )

        index = build_ivfpq_index(
            corpus_embs,
            nlist=args.nlist,
            m=args.m,
            nbits=args.nbits,
            nprobe=args.nprobe,
            train_sample_size=args.train_sample_size,
        )
        id2query = {i: q for i, q in enumerate(corpus)}

        if args.index_path:
            save_index(index, id2query, args.index_path)

    # ── Load test data & per-query search (real-time simulation) ────
    logger.info("Loading test data: %s", args.test_file)
    test_rows = load_test_data(
        args.test_file,
        context_sep=args.context_sep,
        max_ctx=args.max_ctx,
    )
    logger.info("Test examples: %d", len(test_rows))

    logger.info("Searching top-%d candidates (per-query)...", args.top_k)
    results: list[dict] = []
    with InferenceBenchmark(
        project_name="qs_inference",
        track_energy=args.track_energy,
    ) as bench:
        for row in tqdm(test_rows, desc="Inference"):
            pred = bench.timed_call(
                suggest_per_query,
                row["query"],
                query_tokenizer,
                query_model,
                index,
                id2query,
                device,
                top_k=args.top_k,
                max_length=args.max_length,
            )
            results.append(
                {
                    "query": row["query"],
                    "gold_candidate": row["gold_candidate"],
                    "pred_candidates": pred,
                }
            )

    # ── Write results ────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    # Append benchmark stats as last line
    f_stats = os.path.splitext(args.output_file)
    stats_path = f_stats[0] + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(bench.summary(), f, ensure_ascii=False, indent=2)
    logger.info("Stats written to %s", stats_path)

    logger.info("Results written to %s (%d lines)", args.output_file, len(results))


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        ivalue = int(value)
        if ivalue < 1:
            raise argparse.ArgumentTypeError("must be an integer >= 1")
        return ivalue

    p = argparse.ArgumentParser(
        description="Vector search inference for query suggestion"
    )
    # Model paths
    p.add_argument("--query_encoder_path", type=str, required=True)
    p.add_argument("--candidate_encoder_path", type=str, default=None)
    # Data paths
    p.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Candidate corpus file (one query per line)",
    )
    p.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Test JSONL file with context/target",
    )
    p.add_argument("--output_file", type=str, required=True, help="Output JSONL path")
    # Index
    p.add_argument(
        "--index_path", type=str, default=None, help="Path to save/load FAISS index"
    )
    p.add_argument("--overwrite_index", action="store_true")
    # Benchmark
    p.add_argument(
        "--track_energy",
        action="store_true",
        help="Enable energy tracking via codecarbon",
    )
    # Search params
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_length", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--context_sep", type=str, default="[SEP]")
    p.add_argument(
        "--max_ctx",
        type=positive_int,
        default=None,
        help="Use only the newest N context queries",
    )
    # FAISS IVFPQ params
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
