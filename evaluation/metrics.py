from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from bert_score import score as bert_score_fn
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------
class MRRMetric:
    """Mean Reciprocal Rank (exact string match)."""

    def compute(
        self, preds: List[List[str]], gold: List[str]
    ) -> dict[str, float | List[float]]:
        if len(preds) != len(gold):
            raise ValueError(
                f"preds({len(preds)}) and gold({len(gold)}) length mismatch."
            )
        if not preds:
            return {"mean": 0.0, "list": []}

        rr_list: List[float] = []
        for ranked_list, gt in zip(preds, gold):
            try:
                rank = ranked_list.index(gt) + 1
                rr = 1.0 / rank
            except ValueError:
                rr = 0.0
            rr_list.append(rr)

        return {
            "mean": float(np.mean(rr_list)) if rr_list else 0.0,
            "list": rr_list,
        }


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------
class BertScoreMetric:
    """BERTScore similarity — selects best-F1 candidate per sample."""

    def __init__(
        self,
        model_type: str = "ku-nlp/deberta-v2-large-japanese",
        num_layers: int = 24,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.model_type = model_type
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def compute(
        self, preds: List[List[str]], gold: List[str]
    ) -> dict[str, float | List[float]]:
        flat_cands: List[str] = []
        flat_refs: List[str] = []
        group_lengths: List[int] = []

        for cands, ref in zip(preds, gold):
            if not cands:
                group_lengths.append(0)
                continue
            flat_cands.extend(cands)
            flat_refs.extend([ref] * len(cands))
            group_lengths.append(len(cands))

        if not flat_cands:
            zeros = [0.0] * len(preds)
            return {
                "f1": 0.0,
                "p": 0.0,
                "r": 0.0,
                "f1_list": zeros,
                "p_list": zeros,
                "r_list": zeros,
            }

        P, R, F1 = bert_score_fn(
            flat_cands,
            flat_refs,
            model_type=self.model_type,
            num_layers=self.num_layers,
            verbose=False,
            device=self.device,
            batch_size=self.batch_size,
            lang="dummy",
        )

        p_vals = P.detach().cpu().numpy()
        r_vals = R.detach().cpu().numpy()
        f1_vals = F1.detach().cpu().numpy()

        selected_p: List[float] = []
        selected_r: List[float] = []
        selected_f1: List[float] = []
        idx = 0

        for length in group_lengths:
            if length == 0:
                selected_p.append(0.0)
                selected_r.append(0.0)
                selected_f1.append(0.0)
                continue
            end = idx + length
            group_f1 = f1_vals[idx:end]
            best = int(np.argmax(group_f1))
            selected_f1.append(float(f1_vals[idx + best]))
            selected_p.append(float(p_vals[idx + best]))
            selected_r.append(float(r_vals[idx + best]))
            idx = end

        return {
            "f1": float(np.mean(selected_f1)),
            "p": float(np.mean(selected_p)),
            "r": float(np.mean(selected_r)),
            "f1_list": selected_f1,
            "p_list": selected_p,
            "r_list": selected_r,
        }


# ---------------------------------------------------------------------------
# Relevance-Weighted Intra-List Diversity  (R-ILD)
# ---------------------------------------------------------------------------
class RelevanceWeightedILDMetric:
    """
    RW-ILD = Σ(w_i · w_j · dist_ij) / Σ(w_i · w_j)

    Weights are BERTScore F1 (with baseline rescaling).
    Distance is 1 − cosine similarity of mean-pooled embeddings.
    """

    def __init__(
        self,
        ild_model_type: str = "ku-nlp/deberta-v2-large-japanese",
        bert_score_model_type: str = "ku-nlp/deberta-v2-large-japanese",
        baseline_path: str | None = None,
        bert_score_layers: int = 24,
        batch_size: int = 64,
        device: str | None = None,
    ) -> None:
        self.ild_model_type = ild_model_type
        self.bert_score_model_type = bert_score_model_type
        self.bert_score_layers = bert_score_layers
        self.baseline_path = baseline_path
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(ild_model_type)
        self.model = AutoModel.from_pretrained(ild_model_type).to(self.device)
        self.model.eval()

    # -- embedding helper ---------------------------------------------------
    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        all_embs: list[torch.Tensor] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=32,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                out = self.model(**enc)
            tok_emb = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).expand(tok_emb.size()).float()
            mean_emb = torch.sum(tok_emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            all_embs.append(F.normalize(mean_emb, p=2, dim=1))
        if not all_embs:
            return torch.tensor([], device=self.device)
        return torch.cat(all_embs, dim=0)

    # -- main entry ---------------------------------------------------------
    def compute(
        self,
        preds: List[List[str]],
        gold: List[str],
        min_weight: float = 0.0,
    ) -> dict[str, float | List[float]]:
        flat_cands: List[str] = []
        flat_refs: List[str] = []
        group_lengths: List[int] = []

        for cands, ref in zip(preds, gold):
            if not cands:
                group_lengths.append(0)
                continue
            flat_cands.extend(cands)
            flat_refs.extend([ref] * len(cands))
            group_lengths.append(len(cands))

        if not flat_cands:
            return {"rw_ild": 0.0, "rw_ild_list": [0.0] * len(preds)}

        # --- relevance weights via BERTScore ---
        bs_kwargs: dict = dict(
            model_type=self.bert_score_model_type,
            num_layers=self.bert_score_layers,
            verbose=False,
            device=self.device,
            batch_size=self.batch_size,
        )
        if self.baseline_path is not None:
            bs_kwargs.update(
                rescale_with_baseline=True,
                baseline_path=self.baseline_path,
                lang="dummy",
            )
        _, _, F1 = bert_score_fn(flat_cands, flat_refs, **bs_kwargs)
        weights = torch.clamp(F1, min=min_weight).to(self.device)

        # --- embeddings ---
        embeddings = self._get_embeddings(flat_cands)

        # --- per-group RW-ILD ---
        rw_ild_list: List[float] = []
        idx = 0
        for length in group_lengths:
            if length <= 1:
                rw_ild_list.append(0.0)
                idx += length
                continue
            end = idx + length
            emb = embeddings[idx:end]
            w = weights[idx:end]

            # Pairwise cosine distance and relevance-weight outer product
            sim = torch.matmul(emb, emb.T)
            dist = 1.0 - sim
            w_mat = torch.outer(w, w)

            # Exclude diagonal (i == j) to sum over distinct pairs only
            off_diag = ~torch.eye(length, dtype=torch.bool, device=dist.device)
            numerator = torch.sum(w_mat[off_diag] * dist[off_diag])
            denominator = torch.sum(w_mat[off_diag])

            if denominator.item() <= 1e-9:
                rw_ild_list.append(0.0)
            else:
                rw_ild_list.append((numerator / denominator).item())
            idx = end

        return {
            "rw_ild": float(np.mean(rw_ild_list)),
            "rw_ild_list": rw_ild_list,
        }
