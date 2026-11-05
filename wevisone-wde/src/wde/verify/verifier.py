"""VLM / rule verification (§2.2 / §2.3).

Paper: Qwen3-VL verifies query–instance correspondence, dedups, resolves conflicts.
MVP: rule backend (IoU dedup + score accept) + reserved VLM JSON interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

from wde.propose.experts import Proposal
from wde.utils import iou_xyxy


@dataclass
class VerifyDecision:
    proposal_index: int
    decision: str  # accept | reject | uncertain | duplicate
    score: float
    reason: str
    backend: str
    raw: dict[str, Any] = field(default_factory=dict)


class Verifier(ABC):
    name: str

    @abstractmethod
    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
    ) -> list[VerifyDecision]:
        ...


class RuleVerifier(Verifier):
    """IoU NMS-style dedup across experts + score threshold accept."""

    name = "rule"

    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
    ) -> list[VerifyDecision]:
        if not proposals:
            return []

        # Sort by score desc
        order = sorted(range(len(proposals)), key=lambda i: proposals[i].score, reverse=True)
        kept: list[int] = []
        decisions: list[VerifyDecision | None] = [None] * len(proposals)

        for idx in order:
            p = proposals[idx]
            if p.score < accept_min_score:
                decisions[idx] = VerifyDecision(
                    proposal_index=idx,
                    decision="reject",
                    score=p.score,
                    reason=f"score {p.score:.3f} < {accept_min_score}",
                    backend=self.name,
                    raw={"mock_vlm": False},
                )
                continue
            dup_of = None
            for k in kept:
                if iou_xyxy(p.as_xyxy(), proposals[k].as_xyxy()) >= iou_dedup:
                    dup_of = k
                    break
            if dup_of is not None:
                decisions[idx] = VerifyDecision(
                    proposal_index=idx,
                    decision="duplicate",
                    score=p.score,
                    reason=f"IoU>={iou_dedup} with proposal[{dup_of}] ({proposals[dup_of].expert_name})",
                    backend=self.name,
                    raw={"duplicate_of": dup_of},
                )
            else:
                kept.append(idx)
                decisions[idx] = VerifyDecision(
                    proposal_index=idx,
                    decision="accept",
                    score=p.score,
                    reason="passed score + IoU dedup (rule stand-in for Qwen3-VL)",
                    backend=self.name,
                    raw={
                        "note": "MOCK/rule — not Qwen3-VL",
                        "query": query_text,
                        "query_type": query_type,
                    },
                )

        return [d for d in decisions if d is not None]


class VLMJSONVerifier(Verifier):
    """Reserved VLM JSON interface — not wired to a real model in MVP."""

    name = "vlm_json"

    def verify(
        self,
        query_text: str,
        query_type: str,
        proposals: Sequence[Proposal],
        *,
        iou_dedup: float,
        accept_min_score: float,
    ) -> list[VerifyDecision]:
        raise NotImplementedError(
            "VLM JSON verifier reserved for Qwen3-VL. Use backend=rule for MVP."
        )


def build_verifier(name: str) -> Verifier:
    if name == "vlm_json":
        return VLMJSONVerifier()
    return RuleVerifier()
