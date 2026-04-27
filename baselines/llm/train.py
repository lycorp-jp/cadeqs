from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import set_seed
from trl import DataCollatorForCompletionOnlyLM

from baselines.llm.prompt_templates import (
    CHAT_TEMPLATES,
    INSTRUCTION_TEMPLATES,
    RESPONSE_TEMPLATES,
    get_system_prompt,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    base_model: str = field(metadata={"help": "Pretrained causal-LM name or path."})
    lora_r: int = field(default=128, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})


@dataclass
class DataArguments:
    train_file: str = field(metadata={"help": "Path to JSONL training file."})
    dev_file: Optional[str] = field(
        default=None,
        metadata={"help": "Path to JSONL dev file (loaded non-streaming for eval)."},
    )
    max_length: int = field(default=128, metadata={"help": "Max token length."})
    context_sep: str = field(
        default="</s>",
        metadata={"help": "Separator for context queries."},
    )
    lang: str = field(
        default="en",
        metadata={"help": "Template language.", "choices": ["ja", "en"]},
    )
    dump_prompts: int = field(
        default=0,
        metadata={"help": "Print first N prompts to stdout before training."},
    )


@dataclass
class LLMTrainingArguments(TrainingArguments):
    """
    TrainingArguments with FT-LLM defaults reported in the paper.
    """

    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=16)
    gradient_accumulation_steps: int = field(default=8)
    per_device_eval_batch_size: int = field(default=16)
    seed: int = field(default=0)
    learning_rate: float = field(default=1e-3)
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


def build_conversation(
    context: List[str], target: str, sep: str, lang: str
) -> List[Dict[str, str]]:
    """Convert a (context, target) pair into a chat conversation list."""
    # context in JSONL is new→old; reverse to old→new
    joined = sep.join(reversed(context))
    return [
        {"role": "system", "content": get_system_prompt(lang, sep)},
        {"role": "user", "content": joined},
        {"role": "assistant", "content": target},
    ]


def make_tokenize_fn(tokenizer, sep: str, lang: str, max_length: int):
    """Return a batched map function that converts raw rows to token ids."""

    def tokenize_fn(batch: Dict[str, List]):
        texts = []
        for ctx, tgt in zip(batch["context"], batch["target"]):
            conv = build_conversation(ctx, tgt, sep, lang)
            texts.append(
                tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
            )
        return tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    return tokenize_fn


def main() -> None:
    parser = HfArgumentParser((ModelArguments, LLMTrainingArguments, DataArguments))
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    # DDP + LoRA + gradient checkpointing can trigger reentrant-backward issues.
    # Use non-reentrant checkpointing for compatibility with recent PyTorch.
    if getattr(training_args, "gradient_checkpointing", False):
        gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
        if "use_reentrant" not in gc_kwargs:
            gc_kwargs["use_reentrant"] = False
            training_args.gradient_checkpointing_kwargs = gc_kwargs

    # LoRA training under DDP generally expects no unused-parameter search.
    if training_args.ddp_find_unused_parameters is None:
        training_args.ddp_find_unused_parameters = False

    set_seed(training_args.seed)
    is_main = training_args.local_rank <= 0

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CHAT_TEMPLATES[data_args.lang]
    sep = data_args.context_sep

    # ---- Dataset ----
    raw_ds = load_dataset("json", data_files=data_args.train_file, split="train")

    # dump prompts for debugging
    if is_main and data_args.dump_prompts > 0:
        print(f"===== Dumping first {data_args.dump_prompts} prompts =====", flush=True)
        for i, row in enumerate(raw_ds.take(data_args.dump_prompts)):
            conv = build_conversation(
                row["context"], row["target"], sep, data_args.lang
            )
            prompt = tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            print(f"--- sample {i} ---\n{prompt}\n", flush=True)
        print("===========================================\n", flush=True)

    tokenize_fn = make_tokenize_fn(tokenizer, sep, data_args.lang, data_args.max_length)
    train_ds = raw_ds.map(
        tokenize_fn, batched=True, remove_columns=["context", "target"]
    ).shuffle(seed=training_args.seed)

    # dev dataset (non-streaming for eval)
    eval_ds = None
    if data_args.dev_file:
        dev_raw = load_dataset("json", data_files=data_args.dev_file, split="train")
        eval_ds = dev_raw.map(
            tokenize_fn, batched=True, remove_columns=dev_raw.column_names
        )
    elif training_args.eval_strategy != "no":
        logger.warning(
            "No --dev_file specified; disabling evaluation and best-checkpoint selection."
        )
        training_args.eval_strategy = "no"
        training_args.load_best_model_at_end = False

    # ---- Data collator ----
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=INSTRUCTION_TEMPLATES[data_args.lang],
        response_template=RESPONSE_TEMPLATES[data_args.lang],
    )

    # ---- Model (4-bit QLoRA) ----
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        quantization_config=quant_cfg,
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )

    lora_cfg = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_cfg)

    if is_main:
        model.print_trainable_parameters()

    # ---- Training ----
    training_args.remove_unused_columns = False

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()

    # ---- Save: dequantize → merge LoRA → save full-precision model ----
    if is_main:
        _save_merged(model, tokenizer, model_args.base_model, training_args.output_dir)
        logger.info("Done.")


def _save_merged(
    peft_model,
    tokenizer,
    base_model_name: str,
    output_dir: str,
) -> None:
    """Save LoRA adapter, reload base in bf16, merge, and save."""
    tmp_lora = tempfile.mkdtemp(prefix="lora_adapter_")
    try:
        # 1. save adapter weights
        logger.info("Saving LoRA adapter to temp dir …")
        peft_model.save_pretrained(tmp_lora)

        # 2. free quantized model
        del peft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. reload base in full precision on CPU
        logger.info("Reloading base model in bf16 on CPU …")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )

        # 4. attach LoRA and merge
        logger.info("Merging LoRA weights …")
        merged = PeftModel.from_pretrained(base, tmp_lora)
        merged = merged.merge_and_unload()

        # 5. save
        os.makedirs(output_dir, exist_ok=True)
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Saved merged model to %s", output_dir)
    finally:
        shutil.rmtree(tmp_lora, ignore_errors=True)


if __name__ == "__main__":
    main()
