import argparse
import json
import logging
import os
from typing import List

import torch
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer
from tqdm import tqdm

from cadeqs.util import InferenceBenchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(test_path: str, eos_token: str = "</s>") -> List[dict]:
    """Load test JSONL; reverse context (new->old to old->new)
    and join with EOS token."""
    with open(test_path, encoding="utf-8") as f:
        objs = [json.loads(line) for line in f if line.strip()]

    rows: list[dict] = []
    for obj in objs:
        ctx = list(reversed(obj["context"]))
        query = eos_token.join(ctx) if ctx else ""
        rows.append(
            {
                "query": query,
                "gold_candidate": obj["target"],
            }
        )
    return rows


def generate_suggestions(
    prompt: str,
    tokenizer: AutoTokenizer,
    model: ORTModelForSeq2SeqLM,
    *,
    max_length: int = 10,
    num_return_sequences: int = 20,
    top_p: float = 0.90,
    temperature: float = 0.70,
    k: int = 20,
    dedup: bool = False,
) -> List[str]:
    """Generate query suggestions for a single prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
        )

    raw = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    seen: set[str] = set()
    suggestions: list[str] = []
    for s in raw:
        if dedup:
            if s == prompt or s in seen:
                continue
            seen.add(s)
        suggestions.append(s)
        if len(suggestions) >= k:
            break
    return suggestions[:k]


def run(args: argparse.Namespace) -> None:
    logger.info("Loading model: %s (ONNX Runtime)", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = ORTModelForSeq2SeqLM.from_pretrained(
        args.model_path,
        export=True,
        provider=args.ort_provider,
        use_io_binding=False,
    )
    logger.info("ORT provider: %s", args.ort_provider)

    test_rows = load_test_data(args.test_file, eos_token=tokenizer.eos_token)
    logger.info("Test examples: %d", len(test_rows))

    results: list[dict] = []
    with InferenceBenchmark(
        project_name="t5_inference",
        track_energy=args.track_energy,
    ) as bench:
        for row in tqdm(test_rows, desc="Inference"):
            pred = bench.timed_call(
                generate_suggestions,
                row["query"],
                tokenizer,
                model,
                max_length=args.max_length,
                num_return_sequences=args.num_return_sequences,
                top_p=args.top_p,
                temperature=args.temperature,
                k=args.top_k,
                dedup=args.dedup,
            )
            results.append(
                {
                    "query": row["query"],
                    "gold_candidate": row["gold_candidate"],
                    "pred_candidates": pred,
                }
            )

    os.makedirs(
        os.path.dirname(args.output_file) or ".",
        exist_ok=True,
    )
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
    p = argparse.ArgumentParser(description="T5 inference with ONNX Runtime")
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned T5 model",
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
        "--ort_provider",
        type=str,
        default="CUDAExecutionProvider",
        help="ONNX Runtime execution provider (default: GPU)",
    )
    p.add_argument(
        "--track_energy",
        action="store_true",
        help="Enable energy tracking via codecarbon",
    )
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_length", type=int, default=10)
    p.add_argument(
        "--num_return_sequences",
        type=int,
        default=20,
    )
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.70,
    )
    p.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate generated suggestions (default: disabled)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
