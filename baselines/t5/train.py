import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import Dataset, Subset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Pretrained T5 model name or path."},
    )
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a checkpoint to resume training from."},
    )


@dataclass
class DataArguments:
    train_file: str = field(
        metadata={"help": "Path to JSONL training file."},
    )
    max_source_length: int = field(
        default=32,
        metadata={"help": "Max tokenised length for the source (context) side."},
    )
    max_target_length: int = field(
        default=10,
        metadata={"help": "Max tokenised length for the target side."},
    )
    eval_dataset_size: int = field(
        default=10000,
        metadata={"help": "Number of examples held out for evaluation."},
    )


@dataclass
class T5TrainingArguments(Seq2SeqTrainingArguments):
    """
    Seq2SeqTrainingArguments with FT-T5 defaults reported in the paper.
    """

    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=512)
    per_device_eval_batch_size: int = field(default=512)
    seed: int = field(default=0)
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


class QuerySuggestionDataset(Dataset):
    """JSONL dataset: ``{"context": [...], "target": "..."}``."""

    def __init__(self, jsonl_path: str, eos_token: str = "</s>") -> None:
        self.rows: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))
        self.eos_token = eos_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        row = self.rows[idx]
        # Context in JSONL is new → old; reverse to old → new
        # (earliest first), then join with the EOS token.
        ctx = list(reversed(row["context"]))
        source = self.eos_token.join(ctx) if ctx else ""
        return {"source": source, "target": row["target"]}


class Seq2SeqTextPairDataCollator:
    def __init__(
        self,
        tokenizer,
        max_source_length: int = 32,
        max_target_length: int = 10,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict[str, str]]) -> dict:
        sources = [ex["source"] for ex in batch]
        targets = [ex["target"] for ex in batch]

        model_inputs = self.tokenizer(
            sources,
            max_length=self.max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            targets,
            max_length=self.max_target_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


def split_dataset(dataset: Dataset, eval_size: int, seed: int) -> tuple[Subset, Subset]:
    total = len(dataset)
    eval_size = min(eval_size, total // 2)
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    train = Subset(dataset, indices[eval_size:])
    val = Subset(dataset, indices[:eval_size])
    return train, val


def main() -> None:
    parser = HfArgumentParser((ModelArguments, T5TrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # Tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, use_fast=False
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    # Dataset
    dataset = QuerySuggestionDataset(
        data_args.train_file, eos_token=tokenizer.eos_token
    )
    train_dataset, eval_dataset = split_dataset(
        dataset, data_args.eval_dataset_size, seed=training_args.seed
    )

    # Collator
    data_collator = Seq2SeqTextPairDataCollator(
        tokenizer=tokenizer,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
    )

    # Trainer
    training_args.remove_unused_columns = False
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=model_args.checkpoint_path)
    trainer.save_model()

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()
    logger.info(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
