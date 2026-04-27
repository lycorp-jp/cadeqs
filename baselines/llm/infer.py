from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM
from vllm.sampling_params import SamplingParams

from baselines.llm.prompt_templates import (
    CHAT_TEMPLATES,
    RESPONSE_TEMPLATES,
    get_system_prompt,
)
from cadeqs.util import InferenceBenchmark

os.environ.setdefault("VLLM_USE_V1", "0")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(test_path: str, sep: str = "</s>") -> List[dict]:
    """Load test JSONL; reverse context (new->old to old->new)
    and join with separator token."""
    with open(test_path, encoding="utf-8") as f:
        objs = [json.loads(line) for line in f if line.strip()]

    rows: list[dict] = []
    for obj in objs:
        ctx = list(reversed(obj["context"]))
        query = sep.join(ctx) if ctx else ""
        rows.append(
            {
                "query": query,
                "gold_candidate": obj["target"],
            }
        )
    return rows


def build_prompt(
    query: str,
    tokenizer: AutoTokenizer,
    lang: str,
    sep: str,
) -> str:
    """Build a chat-style prompt for generation (no assistant content)."""
    conv = [
        {"role": "system", "content": get_system_prompt(lang, sep)},
        {"role": "user", "content": query},
    ]
    return tokenizer.apply_chat_template(
        conv,
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_suggestions(
    prompt: str,
    model: LLM,
    sampling_params: SamplingParams,
    response_labels: List[str],
    *,
    k: int = 20,
    dedup: bool = False,
) -> List[str]:
    """Generate query suggestions for a single prompt via vLLM."""
    outputs = model.generate([prompt], sampling_params)

    raw_texts = [o.text for o in outputs[0].outputs]

    # Post-process: strip response template prefix and take first line.
    seen: set[str] = set()
    suggestions: list[str] = []
    for text in raw_texts:
        body = text
        for label in response_labels:
            if label in body:
                body = body.split(label, 1)[-1]
                break
        first_line = body.strip().splitlines()
        if not first_line:
            continue
        candidate = first_line[0].strip()
        if not candidate:
            continue
        if dedup:
            if candidate in seen:
                continue
            seen.add(candidate)
        suggestions.append(candidate)
        if len(suggestions) >= k:
            break

    return suggestions[:k]


def run(args: argparse.Namespace) -> None:
    logger.info("Loading tokenizer: %s", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATES[args.lang]

    sep = args.context_sep

    logger.info("Loading vLLM model: %s", args.model_path)
    model = LLM(
        args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        n=args.num_return_sequences,
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["\n"],
    )

    response_labels = [
        RESPONSE_TEMPLATES[args.lang],
        RESPONSE_TEMPLATES["ja"],
        RESPONSE_TEMPLATES["en"],
    ]

    test_rows = load_test_data(test_path=args.test_file, sep=sep)
    logger.info("Test examples: %d", len(test_rows))

    results: list[dict] = []
    with InferenceBenchmark(
        project_name="llm_vllm_inference",
        track_energy=args.track_energy,
    ) as bench:
        for row in tqdm(test_rows, desc="Inference"):
            prompt = build_prompt(row["query"], tokenizer, args.lang, sep)
            pred = bench.timed_call(
                generate_suggestions,
                prompt,
                model,
                sampling_params,
                response_labels,
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
    p = argparse.ArgumentParser(
        description="LLM inference with vLLM",
    )
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged LLM checkpoint (from train.py)",
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
        "--lang",
        type=str,
        default="en",
        choices=["ja", "en"],
        help="Prompt language (default: en)",
    )
    p.add_argument(
        "--track_energy",
        action="store_true",
        help="Enable energy tracking via codecarbon",
    )
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=10)
    p.add_argument(
        "--num_return_sequences",
        type=int,
        default=20,
    )
    p.add_argument("--top_p", type=float, default=0.90)
    p.add_argument("--temperature", type=float, default=0.70)
    p.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate generated suggestions (default: disabled)",
    )
    p.add_argument(
        "--context_sep",
        type=str,
        default="</s>",
        help="Separator for context queries (default: </s>)",
    )
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="vLLM maximum model length",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
