from __future__ import annotations

import argparse
import sys
from typing import Sequence

import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def get_encoder_layers(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer, ("encoder", "layer")
    if hasattr(model, "roberta") and hasattr(model.roberta, "encoder"):
        return model.roberta.encoder.layer, ("roberta", "encoder", "layer")
    if hasattr(model, "transformer") and hasattr(model.transformer, "layer"):
        return model.transformer.layer, ("transformer", "layer")
    if hasattr(model, "distilbert") and hasattr(model.distilbert, "transformer"):
        return model.distilbert.transformer.layer, (
            "distilbert",
            "transformer",
            "layer",
        )
    raise ValueError(f"Unsupported model structure: {type(model).__name__}")


def set_encoder_layers(
    model, attr_path: Sequence[str], new_layers: nn.ModuleList
) -> None:
    obj = model
    for attr in attr_path[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attr_path[-1], new_layers)


def update_num_layers_in_config(model, num_layers: int) -> None:
    if hasattr(model.config, "num_hidden_layers"):
        model.config.num_hidden_layers = num_layers
    if hasattr(model.config, "n_layers"):
        model.config.n_layers = num_layers


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prune BERT-family encoder layers.")
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--layers_to_keep",
        type=int,
        nargs="+",
        default=[0, 4, 5],
        help="Layer indices to keep.",
    )
    p.add_argument("--slow_tokenizer_fallback", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, trust_remote_code=True
        )
    except Exception as e:
        if not args.slow_tokenizer_fallback:
            raise
        print(f"Fast tokenizer failed: {e}", file=sys.stderr)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )

    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    original_layers, attr_path = get_encoder_layers(model)

    num_original = len(original_layers)
    if not args.layers_to_keep:
        raise ValueError("--layers_to_keep must not be empty")
    if max(args.layers_to_keep) >= num_original or min(args.layers_to_keep) < 0:
        raise ValueError(f"--layers_to_keep indices must be in [0, {num_original - 1}]")

    pruned_layers = nn.ModuleList([original_layers[i] for i in args.layers_to_keep])
    set_encoder_layers(model, attr_path, pruned_layers)
    update_num_layers_in_config(model, len(pruned_layers))

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Pruned model saved to: {args.output_dir}")
    print(f"Original layers: {num_original}")
    print(f"Kept layers: {args.layers_to_keep}")


if __name__ == "__main__":
    main()
