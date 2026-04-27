from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from cadeqs.model import DualEncoderModel


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Query encoder model"})
    candidate_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Candidate encoder model (defaults to model_name_or_path)"},
    )
    similarity_type: str = field(default="cosine")
    temp: float = field(default=0.01)
    tie_weights: bool = field(default=False)
    use_same_tower_negatives: bool = field(
        default=True,
        metadata={"help": "Enable same-tower negatives (Eq. 5 terms)."},
    )
    bidirectional: bool = field(
        default=True,
        metadata={"help": "Enable bidirectional in-batch negatives (Eq. 5)."},
    )


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to JSONL training file"})
    max_seq_length: int = field(default=32)
    eval_dataset_size: int = field(default=10000)
    context_sep: Optional[str] = field(
        default=None,
        metadata={"help": "Separator for context list (default: tokenizer sep_token)"},
    )


@dataclass
class CADEQSTrainingArguments(TrainingArguments):
    """
    TrainingArguments with CADE-QS-friendly defaults.
    """

    num_train_epochs: float = field(default=15.0)
    per_device_train_batch_size: int = field(default=512)
    per_device_eval_batch_size: int = field(default=64)
    seed: int = field(default=0)
    learning_rate: float = field(default=1e-4)
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
    remove_unused_columns: bool = field(default=False)
    save_safetensors: bool = field(default=False)

    def __post_init__(self) -> None:
        if self.bf16:
            cuda_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            if not cuda_ok:
                self.bf16 = False
        super().__post_init__()


class QueryPairDataset(Dataset):
    """Read JSONL where each line is {"context": [...], "target": "..."}."""

    def __init__(
        self,
        jsonl_path: str,
        sep_token: str = "[SEP]",
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.sep_token = sep_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        row = self.rows[idx]
        ctx = row["context"]
        joined = self.sep_token.join(ctx) if ctx else ""
        return {"query": joined, "candidate": row["target"]}


class DualEncoderDataCollator:
    """Tokenise query / candidate into separate BatchEncodings."""

    def __init__(
        self,
        query_tokenizer: PreTrainedTokenizerBase,
        candidate_tokenizer: PreTrainedTokenizerBase | None = None,
        max_length: int = 64,
    ) -> None:
        self.query_tokenizer = query_tokenizer
        self.candidate_tokenizer = candidate_tokenizer or query_tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, str]]) -> dict:
        queries = [ex["query"] for ex in batch]
        candidates = [ex["candidate"] for ex in batch]
        query_enc = self.query_tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        candidate_enc = self.candidate_tokenizer(
            candidates,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {"query_inputs": query_enc, "candidate_inputs": candidate_enc}


def split_dataset(dataset: Dataset, eval_size: int, seed: int) -> tuple[Subset, Subset]:
    total = len(dataset)
    eval_size = min(eval_size, total // 2)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[eval_size:]), Subset(dataset, indices[:eval_size])


def main() -> None:
    parser = HfArgumentParser((ModelArguments, CADEQSTrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    # tokenizers
    query_tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    candidate_tok = AutoTokenizer.from_pretrained(
        model_args.candidate_model_name_or_path or model_args.model_name_or_path,
    )

    # dataset
    sep = data_args.context_sep or query_tok.sep_token or "[SEP]"
    ds = QueryPairDataset(data_args.train_file, sep_token=sep)
    train_ds, eval_ds = split_dataset(
        ds, data_args.eval_dataset_size, seed=training_args.seed
    )

    # collator & model
    collator = DualEncoderDataCollator(
        query_tok, candidate_tok, max_length=data_args.max_seq_length
    )
    model = DualEncoderModel(
        model_name_or_path=model_args.model_name_or_path,
        candidate_model_name_or_path=model_args.candidate_model_name_or_path,
        similarity_type=model_args.similarity_type,
        temp=model_args.temp,
        tie_weights=model_args.tie_weights,
        use_same_tower_negatives=model_args.use_same_tower_negatives,
        bidirectional=model_args.bidirectional,
    )

    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()

    # save both encoders
    out = training_args.output_dir
    model.query_encoder.save_pretrained(
        f"{out}/query_encoder", safe_serialization=False
    )
    query_tok.save_pretrained(f"{out}/query_encoder")
    model.candidate_encoder.save_pretrained(
        f"{out}/candidate_encoder", safe_serialization=False
    )
    candidate_tok.save_pretrained(f"{out}/candidate_encoder")


if __name__ == "__main__":
    main()
