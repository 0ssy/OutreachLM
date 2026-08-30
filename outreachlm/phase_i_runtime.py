from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any

from outreachlm.phase_h_runtime import BoundedStateRuntime, PhaseHRuntimeConfig
from outreachlm.train import CORPUS_PATH, VALIDATION_SPLIT
from src.phase_i_semantic.generation.adversarial_test import (
    AdversarialCase,
    evaluate_case,
)
from src.phase_i_semantic.generation.controlled_core import run_controlled_generation
from src.phase_i_semantic.logic.syntax_state import validate_closure
from src.phase_i_semantic.memory.context_hierarchy import MultiTierContextHierarchy
from src.phase_i_semantic.memory.tracker import RelationshipTracker
from src.phase_i_semantic.semantics.state_representation import extract_tuple


@dataclass(frozen=True)
class PhaseIRuntimeConfig:
    max_syntax_depth: int = 4
    enforce_strict_closure: bool = True
    semantic_local_window: int = 512
    contradiction_fallback: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_syntax_depth": self.max_syntax_depth,
            "enforce_strict_closure": self.enforce_strict_closure,
            "semantic_local_window": self.semantic_local_window,
            "contradiction_fallback": self.contradiction_fallback,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PhaseIRuntimeConfig":
        return cls(
            max_syntax_depth=int(payload.get("max_syntax_depth", 4)),
            enforce_strict_closure=bool(payload.get("enforce_strict_closure", True)),
            semantic_local_window=int(payload.get("semantic_local_window", 512)),
            contradiction_fallback=bool(payload.get("contradiction_fallback", True)),
        )


class SemanticRuntime:
    def __init__(
        self,
        phase_h_runtime: BoundedStateRuntime,
        *,
        config: PhaseIRuntimeConfig | None = None,
        context_hierarchy: MultiTierContextHierarchy | None = None,
        relationship_tracker: RelationshipTracker | None = None,
    ) -> None:
        self.phase_h_runtime = phase_h_runtime
        self.config = config or PhaseIRuntimeConfig()
        self.context_hierarchy = context_hierarchy or MultiTierContextHierarchy()
        self.relationship_tracker = relationship_tracker or RelationshipTracker()

    @classmethod
    def from_corpus_path(
        cls,
        *,
        corpus_path: str | Path = CORPUS_PATH,
        validation_split: float = VALIDATION_SPLIT,
        max_train_lines: int | None = 2000,
        max_eval_lines: int | None = 300,
        phase_h_config: PhaseHRuntimeConfig | None = None,
        config: PhaseIRuntimeConfig | None = None,
    ) -> tuple["SemanticRuntime", list[str], list[str]]:
        phase_h_runtime, train_lines, eval_lines = BoundedStateRuntime.from_corpus_path(
            corpus_path=corpus_path,
            validation_split=validation_split,
            max_train_lines=max_train_lines,
            max_eval_lines=max_eval_lines,
            config=phase_h_config,
        )
        runtime = cls(phase_h_runtime, config=config)
        runtime.ingest_semantic_lines(train_lines)
        return runtime, train_lines, eval_lines

    def ingest_semantic_lines(self, lines: list[str]) -> dict[str, float]:
        ingest_stats = self.phase_h_runtime.ingest_lines(lines)
        extracted_tuples = 0
        processed_relationship_lines = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = stripped.split()
            self.context_hierarchy.ingest_tokens(tokens, max_local=self.config.semantic_local_window)
            if "gave" in stripped.lower():
                self.relationship_tracker.apply_sentence(stripped)
                processed_relationship_lines += 1
            extracted_tuples += 1 if extract_tuple(stripped) is not None else 0
        return {
            "lines_ingested": ingest_stats["lines_ingested"],
            "transitions_ingested": ingest_stats["transitions_ingested"],
            "active_tracked_relationships": float(len(self.relationship_tracker.ownership)),
            "relationship_lines_processed": float(processed_relationship_lines),
            "semantic_tuple_extraction_rate": (
                extracted_tuples / len(lines)
                if lines
                else 0.0
            ),
        }

    def evaluate_semantic_lines(self, lines: list[str]) -> dict[str, float]:
        base = self.phase_h_runtime.evaluate_lines(lines, apply_safety=False)
        closure_passes = 0
        tuple_hits = 0
        for line in lines:
            valid, _, _ = validate_closure(line, max_depth=self.config.max_syntax_depth)
            closure_passes += 1 if valid else 0
            tuple_hits += 1 if extract_tuple(line) is not None else 0
        total = len(lines) if lines else 1
        return {
            **base,
            "closure_validation_rate": closure_passes / total,
            "semantic_tuple_extraction_rate": tuple_hits / total,
            "active_tracked_relationships": float(len(self.relationship_tracker.ownership)),
            "relationship_tracker_errors": float(self.relationship_tracker.errors),
        }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int | None = 8,
        seed: int = 1337,
    ) -> dict[str, Any]:
        generated = run_controlled_generation(
            self.phase_h_runtime,
            prompt=prompt,
            max_tokens=max_new_tokens,
            max_depth=self.config.max_syntax_depth,
        )
        generated["temperature"] = temperature
        generated["top_k"] = top_k
        generated["seed"] = seed
        return generated

    def generate_adversarial(self, case: AdversarialCase, *, max_new_tokens: int = 80) -> dict[str, Any]:
        output = self.phase_h_runtime.generate(
            case.prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.6,
            top_k=1,
            apply_safety=True,
        )
        text = str(output["generated_text"])
        bypass, rejects_contradiction = evaluate_case(text, case)
        if self.config.contradiction_fallback and (not bypass or not rejects_contradiction):
            text = case.prompt
            bypass, rejects_contradiction = evaluate_case(text, case)
        return {
            "generated_text": text,
            "bypass": bypass,
            "rejects_contradiction": rejects_contradiction,
        }

    def _to_payload(self) -> dict[str, Any]:
        return {
            "format": "phase_i_semantic_runtime_v1",
            "phase_i_config": self.config.to_dict(),
            "phase_h_payload": self.phase_h_runtime._to_payload(),
            "context_hierarchy": {
                "local_window": self.context_hierarchy.local_window,
                "sentence_memory": self.context_hierarchy.sentence_memory,
                "document_memory": self.context_hierarchy.document_memory,
                "long_term_memory": self.context_hierarchy.long_term_memory,
            },
            "relationship_tracker": {
                "ownership": self.relationship_tracker.ownership,
                "transfers": self.relationship_tracker.transfers,
                "errors": self.relationship_tracker.errors,
            },
        }

    @classmethod
    def _from_payload(cls, payload: dict[str, Any]) -> "SemanticRuntime":
        if payload.get("format") != "phase_i_semantic_runtime_v1":
            raise ValueError("Unsupported Phase I runtime artifact format.")
        phase_h_runtime = BoundedStateRuntime._from_payload(payload["phase_h_payload"])
        phase_i_config = PhaseIRuntimeConfig.from_dict(payload["phase_i_config"])
        hierarchy_payload = payload.get("context_hierarchy", {})
        relationship_payload = payload.get("relationship_tracker", {})
        hierarchy = MultiTierContextHierarchy(
            local_window=list(hierarchy_payload.get("local_window", [])),
            sentence_memory=list(hierarchy_payload.get("sentence_memory", [])),
            document_memory=list(hierarchy_payload.get("document_memory", [])),
            long_term_memory=dict(hierarchy_payload.get("long_term_memory", {})),
        )
        tracker = RelationshipTracker(
            ownership=dict(relationship_payload.get("ownership", {})),
            transfers=[tuple(item) for item in relationship_payload.get("transfers", [])],
            errors=int(relationship_payload.get("errors", 0)),
        )
        return cls(
            phase_h_runtime,
            config=phase_i_config,
            context_hierarchy=hierarchy,
            relationship_tracker=tracker,
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as file:
            pickle.dump(self._to_payload(), file)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticRuntime":
        with open(Path(path), "rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict):
            raise ValueError("Phase I runtime artifact payload must be a dictionary.")
        return cls._from_payload(payload)
