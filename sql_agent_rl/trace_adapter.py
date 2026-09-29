# Copyright (c) Microsoft. All rights reserved.
# Adapted for SQL Agent RL: portable package imports/entrypoints; see THIRD_PARTY_NOTICES.md.
"""Audit selected spans without changing upstream triplets or loss masks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.adapter import TracerTraceToTriplet
from agentlightning.types import Span, Triplet
from .config import require_complete_actions


class AuditedAdapter(TracerTraceToTriplet):
    """Preserve adapter output exactly and record actual node/token selection."""

    def __init__(self, *, agent_match: str, audit_dir: str, save_spans: bool = False):
        super().__init__(agent_match=agent_match)
        self.audit_dir = Path(audit_dir)
        self.save_spans = save_spans
        self.audit_pattern = agent_match

    def adapt(self, source: Sequence[Span] | Sequence[ReadableSpan], /) -> list[Triplet]:
        triplets = super().adapt(source)
        names = [triplet.metadata.get("agent_name") for triplet in triplets]
        if any(not isinstance(name, str) or not re.fullmatch(self.audit_pattern, name) for name in names):
            raise AssertionError(f"Node selection mismatch: {names} for {self.agent_match}")
        if source:
            stored_source = [span for span in source if isinstance(span, Span)]
            if len(stored_source) != len(source):
                raise TypeError("Audit requires the actual store's Span objects")
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            first = stored_source[0]
            record = {
                "rollout_id": first.rollout_id,
                "attempt_id": first.attempt_id,
                "pattern": self.agent_match,
                "all_span_names": [span.name for span in source],
                "selected_node_names": names,
                "selected_nodes": len(triplets),
                "prompt_tokens": sum(len(t.prompt.get("token_ids", [])) for t in triplets),
                "response_tokens": sum(len(t.response.get("token_ids", [])) for t in triplets),
                "empty_selection": not triplets,
            }
            # Adapter/evaluator-side evidence; never placed in model-visible prompts.
            (self.audit_dir / f"{first.rollout_id}_{first.attempt_id}.json").write_text(json.dumps(record, indent=2))
            if self.save_spans:
                (self.audit_dir / f"{first.rollout_id}_{first.attempt_id}.spans.json").write_text(
                    json.dumps([span.model_dump(mode="json") for span in stored_source], ensure_ascii=False)
                )
        return triplets


class CompleteTraceAdapter(AuditedAdapter):
    """Only submit the exact write/rewrite actions actually executed by the agent."""

    def adapt(self, source: Sequence[Span] | Sequence[ReadableSpan], /) -> list[Triplet]:
        triplets = super().adapt(source)
        if not source or not isinstance(source[0], Span):
            raise RuntimeError("Completed rollout must have stored spans")
        first = source[0]
        run = Path(self.audit_dir).parent
        basename = f"{first.rollout_id}_{first.attempt_id}.json"
        model_view = json.loads((run / "model_view" / basename).read_text(encoding="utf-8"))
        evaluator = json.loads((run / "evaluator" / basename).read_text(encoding="utf-8"))
        if model_view["source_id"] != evaluator["source_id"]:
            raise RuntimeError("Model and evaluator task identities differ")
        require_complete_actions(len(model_view["candidates"]), [str(t.metadata.get("agent_name")) for t in triplets])
        return triplets
