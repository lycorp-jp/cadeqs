import json
import logging
import random
import sys
from dataclasses import dataclass, field

import torch
from torch import nn
from datasets import Dataset, Features, Value

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from transformers import HfArgumentParser

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Base model path or name"})


@dataclass
class DataArguments:
    train_data: str = field(metadata={"help": "Train JSONL path"})
    eval_dataset_size: int = field(
        default=10000,
        metadata={"help": "Number of examples held out for evaluation."},
    )


@dataclass
class MONRTrainingArguments(SentenceTransformerTrainingArguments):
    """
    SentenceTransformerTrainingArguments with MONR defaults from the paper.
    """

    num_train_epochs: float = field(default=15.0)
    per_device_train_batch_size: int = field(default=128)
    per_device_eval_batch_size: int = field(default=128)
    gradient_accumulation_steps: int = field(default=1)
    seed: int = field(default=0)
    learning_rate: float = field(default=2e-5)
    bf16: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: int = field(default=4000)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=4000)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)
    logging_steps: int = field(default=4000)

    def __post_init__(self) -> None:
        if self.bf16:
            cuda_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if not cuda_ok:
                self.bf16 = False
        super().__post_init__()


class PairwiseCrossEntropyLoss(nn.Module):
    """Pairwise cross-entropy loss on <anchor, positive, negative> triples
    as described in the MONR paper."""

    def __init__(self, model: SentenceTransformer):
        super().__init__()
        self.model = model
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features, labels):
        reps = [self.model(f)["sentence_embedding"] for f in sentence_features]

        rep_anchor = torch.nn.functional.normalize(reps[0], p=2, dim=1)
        rep_pos = torch.nn.functional.normalize(reps[1], p=2, dim=1)
        rep_neg = torch.nn.functional.normalize(reps[2], p=2, dim=1)

        score_pos = torch.sum(rep_anchor * rep_pos, dim=1, keepdim=True)
        score_neg = torch.sum(rep_anchor * rep_neg, dim=1, keepdim=True)

        scores = torch.cat([score_pos, score_neg], dim=1)
        targets = torch.zeros(scores.size(0), dtype=torch.long, device=scores.device)

        return self.cross_entropy_loss(scores, targets)


def load_triplets_from_jsonl(file_path: str, seed: int = 0) -> Dataset:
    """Load JSONL and build anchor/positive/negative triplets.

    Each JSONL line: ``{"context": ["most_recent", ...], "target": "..."}``.
    anchor = context[0] (most recent query), positive = target,
    negative = randomly sampled target from another example.
    """
    logger.info(f"Loading triplets from {file_path} ...")

    rows: list[dict] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not obj.get("context"):
                continue
            rows.append(obj)

    if len(rows) < 2:
        raise ValueError(f"Need at least 2 valid rows, got {len(rows)}")

    targets = [r["target"] for r in rows]
    rng = random.Random(seed)

    anchors, positives, negatives = [], [], []
    for i, row in enumerate(rows):
        anchor = row["context"][0]
        pos = row["target"]

        # Sample a negative target different from the positive
        neg = pos
        for _ in range(100):
            j = rng.randrange(len(targets))
            if targets[j] != pos:
                neg = targets[j]
                break

        anchors.append(anchor)
        positives.append(pos)
        negatives.append(neg)

    features = Features(
        {
            "anchor": Value("string"),
            "positive": Value("string"),
            "negative": Value("string"),
        }
    )
    ds = Dataset.from_dict(
        {"anchor": anchors, "positive": positives, "negative": negatives},
        features=features,
    )
    logger.info(f"Loaded {len(ds)} triplets.")
    return ds


def split_dataset(
    dataset: Dataset, eval_size: int, seed: int
) -> tuple[Dataset, Dataset]:
    total = len(dataset)
    eval_size = min(eval_size, total // 2)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return (
        dataset.select(indices[eval_size:]),
        dataset.select(indices[:eval_size]),
    )


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, MONRTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Model
    logger.info(f"Initializing model: {model_args.model_name_or_path}")
    model = SentenceTransformer(model_args.model_name_or_path, trust_remote_code=True)

    # Datasets
    full_dataset = load_triplets_from_jsonl(
        data_args.train_data, seed=training_args.seed
    )
    train_dataset, eval_dataset = split_dataset(
        full_dataset,
        data_args.eval_dataset_size,
        seed=training_args.seed,
    )

    # Loss
    logger.info("Initializing PairwiseCrossEntropyLoss.")
    loss = PairwiseCrossEntropyLoss(model=model)

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info(f"Saving best model to {training_args.output_dir}...")
    trainer.save_model(training_args.output_dir)
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
