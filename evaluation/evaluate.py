from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from evaluation.metrics import BertScoreMetric, MRRMetric, RelevanceWeightedILDMetric

AVAILABLE_METRICS = ("mrr", "bert_score", "r_ild")


def load_predictions(path: str) -> tuple[List[List[str]], List[str]]:
    preds: List[List[str]] = []
    gold: List[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            preds.append(obj["pred_candidates"])
            gold.append(obj["gold_candidate"])
    return preds, gold


def run(args: argparse.Namespace) -> dict[str, float]:
    preds, gold = load_predictions(args.input)
    print(f"Loaded {len(preds)} samples from {args.input}")

    results: dict = {}

    # Record evaluator model when BERT-based metrics are requested
    if {"bert_score", "r_ild"} & set(args.metrics):
        results["bert_evaluator"] = args.bert_evaluator

    if "mrr" in args.metrics:
        print("Computing MRR ...")
        mrr = MRRMetric().compute(preds, gold)
        results["mrr"] = mrr["mean"]

    if "bert_score" in args.metrics:
        print("Computing BERTScore ...")
        bs = BertScoreMetric(
            model_type=args.bert_evaluator,
            num_layers=args.bert_num_layers,
            batch_size=args.batch_size,
            device=args.device,
        ).compute(preds, gold)
        results["bert_score"] = bs["f1"]

    if "r_ild" in args.metrics:
        print("Computing R-ILD ...")
        rild = RelevanceWeightedILDMetric(
            ild_model_type=args.bert_evaluator,
            bert_score_model_type=args.bert_evaluator,
            baseline_path=args.baseline_path,
            bert_score_layers=args.bert_num_layers,
            batch_size=args.batch_size,
            device=args.device,
        ).compute(preds, gold)
        results["r_ild"] = rild["rw_ild"]

    # Round numeric values to 4 decimal places
    results = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in results.items()
    }

    # --- save results ---
    output_path = args.output
    if output_path is None:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_eval{ext}"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(results, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate query suggestion predictions."
    )
    parser.add_argument("--input", required=True, help="Path to predictions JSONL.")
    parser.add_argument("--output", default=None, help="Path to output results JSONL.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=AVAILABLE_METRICS,
        default=list(AVAILABLE_METRICS),
        help="Metrics to compute (default: all).",
    )
    parser.add_argument(
        "--bert-evaluator",
        default="microsoft/deberta-large",
        help="BERT model for BERTScore and R-ILD.",
    )
    parser.add_argument(
        "--bert-num-layers", type=int, default=24, help="Number of BERT layers to use."
    )
    parser.add_argument(
        "--baseline-path",
        default=None,
        help="BERTScore baseline TSV for R-ILD rescaling.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default=None, help="Device (cuda/cpu). Auto-detected if omitted."
    )

    run(parser.parse_args())


if __name__ == "__main__":
    main()
