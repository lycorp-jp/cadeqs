from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BatchEncoding
from transformers.utils import ModelOutput


def pool_cls(hidden: torch.Tensor, **_: object) -> torch.Tensor:
    """CLS pooling: take the first token representation."""
    return hidden[:, 0]


class Similarity(nn.Module):
    def __init__(self, similarity_type: str = "cosine", temp: float = 0.05) -> None:
        super().__init__()
        self.similarity_type = similarity_type
        self.temp = temp

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.similarity_type == "cosine":
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
        elif self.similarity_type != "dot":
            raise ValueError(f"Unknown similarity_type: {self.similarity_type}")
        return torch.matmul(x, y.T) / self.temp


# ── DDP all-gather with gradient pass-through ────────────────────────


def gather_with_grad(t: torch.Tensor) -> torch.Tensor:
    """All-gather that keeps autograd on the local shard."""
    if not torch.distributed.is_initialized():
        return t
    t = t.contiguous()
    ws = torch.distributed.get_world_size()
    gathered = [torch.zeros_like(t) for _ in range(ws)]
    torch.distributed.all_gather(gathered, t)
    gathered[torch.distributed.get_rank()] = t  # keep gradient
    return torch.cat(gathered, dim=0)


class DualEncoderModel(nn.Module):
    """Dual-encoder (query / candidate) with optional Siamese weights,
    Same-Tower Negatives, and bidirectional (symmetric) InfoNCE loss."""

    def __init__(
        self,
        model_name_or_path: str,
        candidate_model_name_or_path: str | None = None,
        similarity_type: str = "cosine",
        temp: float = 0.05,
        tie_weights: bool = False,
        use_same_tower_negatives: bool = False,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name_or_path)
        if tie_weights:
            self.candidate_encoder = self.query_encoder
        else:
            self.candidate_encoder = AutoModel.from_pretrained(
                candidate_model_name_or_path or model_name_or_path
            )

        self.similarity_fn = Similarity(similarity_type, temp)
        self.loss_fn = nn.CrossEntropyLoss()
        self.tie_weights = tie_weights
        # Same Tower Negatives: https://aclanthology.org/2023.findings-acl.761.pdf
        self.use_stn = use_same_tower_negatives
        self.bidirectional = bidirectional

    # ── helpers ───────────────────────────────────────────────────────

    def _encode(self, model: AutoModel, batch: BatchEncoding) -> torch.Tensor:
        out = model(**batch, return_dict=True)
        return pool_cls(out.last_hidden_state)

    def _build_logits(self, anchor: torch.Tensor, others: torch.Tensor) -> torch.Tensor:
        """Similarity matrix with optional Same-Tower Negatives."""
        sim_cross = self.similarity_fn(anchor, others)  # (B, B')
        if self.use_stn:
            sim_same = self.similarity_fn(anchor, anchor)  # (B, B)
            diag_mask = torch.eye(
                sim_same.size(0), dtype=torch.bool, device=sim_same.device
            )
            sim_same.masked_fill_(diag_mask, -1e4)
            return torch.cat([sim_cross, sim_same], dim=1)  # (B, B'+B)
        return sim_cross

    # ── forward ──────────────────────────────────────────────────────

    def forward(
        self,
        query_inputs: BatchEncoding,
        candidate_inputs: BatchEncoding,
        return_loss: bool = True,
    ) -> ModelOutput:
        q = self._encode(self.query_encoder, query_inputs)
        p = self._encode(self.candidate_encoder, candidate_inputs)

        if self.training:
            q = gather_with_grad(q)
            p = gather_with_grad(p)

        # query → candidate
        logits_q = self._build_logits(q, p)
        labels = torch.arange(logits_q.size(0), device=logits_q.device)
        loss_q = self.loss_fn(logits_q, labels)

        total_loss = loss_q

        # candidate → query (symmetric)
        if self.bidirectional:
            logits_p = self._build_logits(p, q)
            labels_p = torch.arange(logits_p.size(0), device=logits_p.device)
            total_loss = total_loss + self.loss_fn(logits_p, labels_p)

        denom = 2 if self.bidirectional else 1
        loss = total_loss / denom if return_loss else None

        return ModelOutput(loss=loss, logits=logits_q)
