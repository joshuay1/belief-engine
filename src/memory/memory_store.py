from typing import Any, Dict, List, Optional
import copy
import hashlib
import json
import math
from .argument_record import ArgumentRecord

from .merging import merge_arguments
from .retrieval import retrieve_arguments
from .credence_manager import CredenceManager
from ..constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
from ..utils.output import print_memory, print_debug, print_info, get_verbosity
from ..simulation.belief_engine import (
    ACTIVE_SET_BELIEF_SEMANTICS_MODES,
    BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1,
    BELIEF_SEMANTICS_INCREMENTAL_V1,
    create_belief_engine,
)

DYNAMIC_EVIDENCE_HORIZON_PERSISTENT = "persistent"
DYNAMIC_EVIDENCE_HORIZON_LATEST_OPPONENT_TURN = "latest_opponent_turn"
DYNAMIC_EVIDENCE_HORIZONS = (
    DYNAMIC_EVIDENCE_HORIZON_PERSISTENT,
    DYNAMIC_EVIDENCE_HORIZON_LATEST_OPPONENT_TURN,
)


class MemoryStore:
    """
    Manages the storage, retrieval, and updates of arguments.
    """
    def __init__(self, storage_mode: str = "episodic", retrieval_strategy: str = "weighted", worldview_openness: str = "balanced", stance_method: str = "linear", update_sensitivity: float = 2.0, anchor_sensitivity: float = None, argument_similarity_threshold: float = 0.7, self_similarity_threshold: float = 0.95, confirmation_bias: float = 0.0, cross_polarity_dedup: bool = False, deduplication_enabled: bool = True, allow_self_arguments: bool = True, belief_engine_type: str = "bayesian", belief_semantics_mode: str = BELIEF_SEMANTICS_INCREMENTAL_V1, dynamic_evidence_horizon: str = DYNAMIC_EVIDENCE_HORIZON_PERSISTENT, **kwargs):
        self.arguments: Dict[str, ArgumentRecord] = {}
        self.storage_mode = storage_mode
        self.retrieval_strategy = retrieval_strategy
        self.worldview_openness = worldview_openness
        self.stance_method = stance_method
        self.update_sensitivity = update_sensitivity
        self.anchor_sensitivity = anchor_sensitivity
        self.confirmation_bias = confirmation_bias
        self.belief_engine_type = belief_engine_type
        self.belief_semantics_mode = belief_semantics_mode
        self.argument_similarity_threshold = argument_similarity_threshold
        self.self_similarity_threshold = self_similarity_threshold
        self.cross_polarity_dedup = cross_polarity_dedup
        self.deduplication_enabled = bool(deduplication_enabled)
        self.allow_self_arguments = allow_self_arguments
        self.dynamic_evidence_horizon = str(dynamic_evidence_horizon)
        self.agent_id = kwargs.get("agent_id", "Unknown")
        self.backend = kwargs.get("backend")
        self.initial_stance = kwargs.get("initial_stance", 0.0)
        if self.dynamic_evidence_horizon not in DYNAMIC_EVIDENCE_HORIZONS:
            raise ValueError(
                "dynamic_evidence_horizon must be one of "
                f"{DYNAMIC_EVIDENCE_HORIZONS}, got "
                f"{self.dynamic_evidence_horizon!r}"
            )
        if (
            self.dynamic_evidence_horizon
            == DYNAMIC_EVIDENCE_HORIZON_LATEST_OPPONENT_TURN
            and self.belief_semantics_mode
            != BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1
        ):
            raise ValueError(
                "dynamic_evidence_horizon='latest_opponent_turn' requires "
                "belief_semantics_mode='active_explicit_prior_v1'"
            )
        if self.belief_semantics_mode in ACTIVE_SET_BELIEF_SEMANTICS_MODES:
            if self.stance_method != "bayesian":
                raise ValueError(
                    f"{self.belief_semantics_mode} requires stance_method='bayesian'"
                )
            if self.belief_engine_type != "bayesian":
                raise ValueError(
                    f"{self.belief_semantics_mode} requires "
                    "belief_engine_type='bayesian'"
                )
            if self.storage_mode == "agentic":
                raise ValueError(
                    f"{self.belief_semantics_mode} is not implemented for "
                    "AgenticMemoryStore"
                )
        
        # Initialize Credence Manager (still used by VectorStrategy)
        self.credence_manager = CredenceManager(similarity_threshold=argument_similarity_threshold)
        
        # Initialize Judgement Strategy
        from .judgement_strategies import (
            AgenticJudgementStrategy,
            JudgementStrategy,
            NoDedupJudgementStrategy,
            VectorJudgementStrategy,
        )
        
        self.judgement_strategy: JudgementStrategy
        self.judgement_strategy: JudgementStrategy
        # FIX: Check openness directly. Previously restricted to 'worldview' storage, causing 'agentic_episodic' to fallback to Vector.
        if not self.deduplication_enabled:
            print_memory(self.agent_id, "Deduplication disabled")
            self.judgement_strategy = NoDedupJudgementStrategy()
        elif self.worldview_openness == "agentic":
            print_memory(self.agent_id, "Using Agentic Judgement Strategy (Logic-Based)")
            self.judgement_strategy = AgenticJudgementStrategy()
        else:
            cross_pol_str = " + Cross-Polarity" if cross_polarity_dedup else ""
            print_memory(self.agent_id, f"Using Vector Judgement Strategy (Threshold: {argument_similarity_threshold}, Self-Threshold: {self_similarity_threshold}{cross_pol_str})")
            self.judgement_strategy = VectorJudgementStrategy(
                similarity_threshold=argument_similarity_threshold,
                self_similarity_threshold=self_similarity_threshold,
                cross_polarity_dedup=cross_polarity_dedup
            )

        
        # Unified Belief Engine Integration
        # We now use the standardized BeliefEngine for all stance calculations, replacing the old ad-hoc math.
        
        # MemoryStore doesn't know the exact memory_dir, but we can default or get it from kwargs if passed.
        # For SubjectAgent, this might be ephemeral or passed down. 
        # For now, we'll try to use a local path if not provided.
        memory_dir = kwargs.get("memory_dir", f"data/memory/{self.agent_id}")
        
        self.belief_engine = create_belief_engine(
            agent_name=self.agent_id,
            memory_dir=memory_dir,
            config={
                "update_sensitivity": self.update_sensitivity,
                "anchor_sensitivity": self.anchor_sensitivity,
                "confirmation_bias": self.confirmation_bias,
                "belief_engine_type": self.belief_engine_type,
                "belief_semantics_mode": self.belief_semantics_mode,
                "initial_stance": (
                    self.initial_stance
                    if self.belief_semantics_mode
                    == BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1
                    else 0.0
                ),
            }
        )
        self._evidence_horizon_events: List[Dict[str, Any]] = []
        # The turn-local ablation must remove *all* cross-turn influence,
        # including in-place mutations that deduplication can make to seed
        # records. Capture the complete seed-record state at the first
        # receiver turn and restore deep copies on every reset.
        self._seed_state_baseline: Optional[Dict[str, ArgumentRecord]] = None
        self._no_dedup_collision_count = 0
        self._dedup_status_counts: Dict[str, int] = {}
        self._dedup_dynamic_status_counts: Dict[str, int] = {}
        self._dedup_seed_status_counts: Dict[str, int] = {}

    def _make_no_dedup_record_id_unique(
        self,
        argument: ArgumentRecord,
    ) -> None:
        """Prevent exact repeated claims from overwriting earlier occurrences."""
        if self.deduplication_enabled or argument.id not in self.arguments:
            return
        base_id = str(argument.id)
        suffix = 2
        candidate = f"{base_id}::occurrence:{suffix}"
        while candidate in self.arguments:
            suffix += 1
            candidate = f"{base_id}::occurrence:{suffix}"
        argument.id = candidate
        self._no_dedup_collision_count += 1

    def deduplication_audit_summary(
        self,
        include_manipulation_telemetry: bool = True,
    ) -> Dict[str, Any]:
        dynamic_records = {
            argument_id: argument
            for argument_id, argument in self.arguments.items()
            if getattr(argument, "source_role", "") != "seed"
        }
        active_dynamic_records = {
            argument_id: argument
            for argument_id, argument in dynamic_records.items()
            if bool(getattr(argument, "credence_relevant", False))
        }
        archived_dynamic_records = {
            argument_id: argument
            for argument_id, argument in dynamic_records.items()
            if not bool(getattr(argument, "credence_relevant", False))
        }
        summary = {
            "deduplication_enabled": self.deduplication_enabled,
            "status_counts": dict(self._dedup_status_counts),
            "dynamic_status_counts": dict(
                self._dedup_dynamic_status_counts
            ),
            "seed_status_counts": dict(self._dedup_seed_status_counts),
            "no_dedup_id_collisions_preserved": self._no_dedup_collision_count,
            "final_dynamic_records_total": len(dynamic_records),
            "final_dynamic_active_records": len(active_dynamic_records),
            "final_dynamic_archived_records": len(archived_dynamic_records),
        }
        if include_manipulation_telemetry:
            summary.update(
                {
                    "final_dynamic_state_sha256": (
                        self._record_state_fingerprint(dynamic_records)
                    ),
                    "final_dynamic_active_state_sha256": (
                        self._record_state_fingerprint(
                            active_dynamic_records
                        )
                    ),
                    "final_dynamic_canonical_state_sha256": (
                        self._canonical_record_state_fingerprint(
                            dynamic_records
                        )
                    ),
                    "final_dynamic_active_canonical_state_sha256": (
                        self._canonical_record_state_fingerprint(
                            active_dynamic_records
                        )
                    ),
                }
            )
        return summary

    @staticmethod
    def _record_state_fingerprint(
        records: Dict[str, ArgumentRecord],
    ) -> str:
        """Hash complete record state without exposing claims in telemetry."""
        payload = {
            str(argument_id): vars(argument)
            for argument_id, argument in sorted(records.items())
        }
        encoded = json.dumps(
            payload,
            default=str,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _canonical_record_state_fingerprint(
        records: Dict[str, ArgumentRecord],
    ) -> str:
        """Hash intervention-relevant state without IDs or wall-clock fields.

        Python-generated record IDs and timestamps are intentionally excluded,
        so paired cells can be compared across separate worker processes.
        Claims are represented only by SHA-256 fingerprints in the telemetry.
        Duplicate occurrences remain separate list entries.
        """
        payload = []
        for argument in records.values():
            claim_sha256 = hashlib.sha256(
                str(argument.claim).encode("utf-8")
            ).hexdigest()
            payload.append(
                {
                    "claim_sha256": claim_sha256,
                    "polarity": str(argument.polarity),
                    "source_role": str(argument.source_role),
                    "support_strength": float(argument.support_strength),
                    "support_strength_llm": float(
                        argument.support_strength_llm
                    ),
                    "support_strength_classifier": float(
                        argument.support_strength_classifier
                    ),
                    "credence_relevant": bool(
                        argument.credence_relevant
                    ),
                }
            )
        encoded = json.dumps(
            sorted(
                payload,
                key=lambda item: json.dumps(
                    item,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _current_seed_records(self) -> Dict[str, ArgumentRecord]:
        return {
            argument_id: argument
            for argument_id, argument in self.arguments.items()
            if getattr(argument, "source_role", "") == "seed"
        }

    def _restore_seed_state_baseline(self) -> Dict[str, Any]:
        """Restore the exact pre-dynamic seed state for a turn-local reset."""
        current_seed_records = self._current_seed_records()
        if self._seed_state_baseline is None:
            self._seed_state_baseline = copy.deepcopy(current_seed_records)

        baseline = self._seed_state_baseline
        baseline_ids = set(baseline)
        unexpected_seed_ids = set(current_seed_records) - baseline_ids
        if unexpected_seed_ids:
            raise RuntimeError(
                "Seed-record membership changed after the turn-local baseline "
                "was captured."
            )

        before_fingerprint = self._record_state_fingerprint(
            current_seed_records
        )
        baseline_fingerprint = self._record_state_fingerprint(baseline)
        restored_records = sum(
            argument_id not in current_seed_records
            or self._record_state_fingerprint(
                {argument_id: current_seed_records[argument_id]}
            )
            != self._record_state_fingerprint(
                {argument_id: baseline[argument_id]}
            )
            for argument_id in baseline
        )

        # Replace the entire active store with deep copies of the pristine
        # seeds. This restores every mutable dataclass field and also repairs a
        # dynamic record that may have collided with a seed's dictionary key.
        self.arguments.clear()
        self.arguments.update(copy.deepcopy(baseline))
        after_fingerprint = self._record_state_fingerprint(
            self._current_seed_records()
        )
        return {
            "baseline_fingerprint": baseline_fingerprint,
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": after_fingerprint,
            "baseline_records": len(baseline),
            "restored_records": restored_records,
            "restoration_ok": after_fingerprint == baseline_fingerprint,
        }

    def _memory_role_counts(self) -> Dict[str, int]:
        seed_records = sum(
            getattr(argument, "source_role", "") == "seed"
            for argument in self.arguments.values()
        )
        return {
            "total": len(self.arguments),
            "seed": seed_records,
            "dynamic": len(self.arguments) - seed_records,
        }

    def prepare_for_opponent_evidence_turn(
        self,
        *,
        round_index: int,
        sender_name: str,
    ) -> int:
        """Apply the configured horizon before ingesting one opponent message."""
        if (
            self._evidence_horizon_events
            and not self._evidence_horizon_events[-1].get("finalized", False)
        ):
            raise RuntimeError(
                "Previous opponent evidence turn was not finalized before "
                "the next turn began."
            )

        before = self._memory_role_counts()
        stance_before = self.get_stance(verbose=False)
        evicted_ids: List[str] = []
        prior_reconstruction_error = 0.0
        seed_restoration = {
            "baseline_fingerprint": "",
            "before_fingerprint": "",
            "after_fingerprint": "",
            "baseline_records": before["seed"],
            "restored_records": 0,
            "restoration_ok": True,
        }
        reset_applied = (
            self.dynamic_evidence_horizon
            == DYNAMIC_EVIDENCE_HORIZON_LATEST_OPPONENT_TURN
        )

        if reset_applied:
            evicted_ids = [
                argument_id
                for argument_id, argument in self.arguments.items()
                if getattr(argument, "source_role", "") != "seed"
            ]
            seed_restoration = self._restore_seed_state_baseline()
            if not seed_restoration["restoration_ok"]:
                raise RuntimeError(
                    "Turn-local seed-state restoration fingerprint mismatch."
                )
            self.belief_engine.recompute_from_active_records(
                self.arguments.values(),
                persist=True,
                increment_history=False,
            )
            expected_prior_stance = math.tanh(
                float(self.belief_engine.explicit_prior_log_odds) / 2.0
            )
            prior_reconstruction_error = abs(
                self.get_stance(verbose=False) - expected_prior_stance
            )

        after_reset = self._memory_role_counts()
        event = {
            "event_index": len(self._evidence_horizon_events),
            "round": int(round_index),
            "sender": str(sender_name),
            "horizon": self.dynamic_evidence_horizon,
            "reset_applied": reset_applied,
            "before": before,
            "after_reset": after_reset,
            "evicted_records": len(evicted_ids),
            "seed_records_preserved": (
                after_reset["seed"] == seed_restoration["baseline_records"]
                and bool(seed_restoration["restoration_ok"])
            ),
            "seed_state_baseline_fingerprint": seed_restoration[
                "baseline_fingerprint"
            ],
            "seed_state_before_reset_fingerprint": seed_restoration[
                "before_fingerprint"
            ],
            "seed_state_after_reset_fingerprint": seed_restoration[
                "after_fingerprint"
            ],
            "seed_state_restored_records": seed_restoration[
                "restored_records"
            ],
            "seed_state_restoration_ok": seed_restoration[
                "restoration_ok"
            ],
            "stance_before": stance_before,
            "stance_after_reset": self.get_stance(verbose=False),
            "prior_reconstruction_error": prior_reconstruction_error,
            "finalized": False,
        }
        self._evidence_horizon_events.append(event)
        return int(event["event_index"])

    def finalize_opponent_evidence_turn(
        self,
        event_index: int,
        *,
        extracted_records: int,
        accepted_records: int,
        status_counts: Optional[Dict[str, int]] = None,
    ) -> None:
        """Close one receiver-side horizon event after batch ingestion."""
        if event_index < 0 or event_index >= len(self._evidence_horizon_events):
            raise IndexError(f"Unknown evidence-horizon event {event_index}.")
        event = self._evidence_horizon_events[event_index]
        if event.get("finalized", False):
            raise RuntimeError(
                f"Evidence-horizon event {event_index} was already finalized."
            )

        after_batch = self._memory_role_counts()
        active_state_error = (
            abs(
                float(self.belief_engine.log_odds)
                - float(
                    self.belief_engine.compute_active_log_odds(
                        self.arguments.values()
                    )
                )
            )
            if self.belief_semantics_mode in ACTIVE_SET_BELIEF_SEMANTICS_MODES
            else 0.0
        )
        event.update(
            {
                "extracted_records": int(extracted_records),
                "accepted_records": int(accepted_records),
                "status_counts": dict(status_counts or {}),
                "after_batch": after_batch,
                "latest_turn_bound_ok": (
                    self.dynamic_evidence_horizon
                    != DYNAMIC_EVIDENCE_HORIZON_LATEST_OPPONENT_TURN
                    or after_batch["dynamic"] <= int(extracted_records)
                ),
                "active_state_recomputation_error": active_state_error,
                "finalized": True,
            }
        )

    def abort_opponent_evidence_turn(
        self,
        event_index: int,
        *,
        reason: str,
    ) -> None:
        """Close an incomplete horizon event when its turn fails."""
        if event_index < 0 or event_index >= len(self._evidence_horizon_events):
            raise IndexError(f"Unknown evidence-horizon event {event_index}.")
        event = self._evidence_horizon_events[event_index]
        if event.get("finalized", False):
            return
        event.update(
            aborted=True,
            abort_reason=str(reason),
            finalized=True,
        )

    def evidence_horizon_audit_summary(self) -> Dict[str, Any]:
        """Return compact counts plus per-turn records for horizon verification."""
        events = [dict(event) for event in self._evidence_horizon_events]
        completed = [
            event
            for event in events
            if event.get("finalized", False)
            and not event.get("aborted", False)
        ]
        def total(key, source=events):
            return sum(int(event.get(key, 0)) for event in source)

        def maximum(key, source):
            return max(
                (float(event.get(key, 0.0)) for event in source),
                default=0.0,
            )
        return {
            "dynamic_evidence_horizon": self.dynamic_evidence_horizon,
            "prepare_calls": len(events),
            "completed_turns": len(completed),
            "aborted_turns": sum(event.get("aborted", False) for event in events),
            "open_turns": sum(not event.get("finalized", False) for event in events),
            "reset_calls": sum(event.get("reset_applied", False) for event in events),
            "evicted_records": total("evicted_records"),
            "seed_state_restored_records": total(
                "seed_state_restored_records"
            ),
            "extracted_records": total("extracted_records", completed),
            "accepted_records": total("accepted_records", completed),
            "prior_reconstruction_error_max": maximum(
                "prior_reconstruction_error",
                events,
            ),
            "active_state_recomputation_error_max": maximum(
                "active_state_recomputation_error",
                completed,
            ),
            "post_batch_dynamic_records_max": max(
                (
                    int(event.get("after_batch", {}).get("dynamic", 0))
                    for event in completed
                ),
                default=0,
            ),
            "all_seed_records_preserved": all(
                bool(event.get("seed_records_preserved", False))
                for event in events
            ),
            "all_seed_state_restorations_ok": all(
                bool(event.get("seed_state_restoration_ok", False))
                for event in events
            ),
            "all_latest_turn_bounds_ok": all(
                bool(event.get("latest_turn_bound_ok", False))
                for event in completed
            ),
            "events": events,
        }

    def _add_argument_internal(
        self,
        argument: ArgumentRecord,
        backend=None,
        persist_belief_state: bool = True,
        trace_memory: bool = True
    ) -> (str, str, str, ArgumentRecord, bool):
        """
        Adds an argument to the store using the configured Evolution Strategy.
        Returns (status, reason, snippet, record, belief_updated).
        """
        if not self.allow_self_arguments and getattr(argument, "source_role", "") == "self":
            snippet = argument.claim[:60] + "..."
            return "ignored", "Self arguments disabled", snippet, argument, False

        self._make_no_dedup_record_id_unique(argument)
        snippet = argument.claim[:60] + "..."
        belief_updated = False

        # Delegate entirely to the Strategy
        # The strategy modifies flags (credence_relevant) in-place
        # 2. Delegate to Judgement Strategy (Gatekeeper)
        status, target_id, reason, processed_arg = self.judgement_strategy.process(argument, self.arguments, backend)
        self._dedup_status_counts[status] = (
            self._dedup_status_counts.get(status, 0) + 1
        )
        role_counts = (
            self._dedup_seed_status_counts
            if getattr(argument, "source_role", "") == "seed"
            else self._dedup_dynamic_status_counts
        )
        role_counts[status] = role_counts.get(status, 0) + 1
        print_debug(f"add_argument processed '{argument.claim[:20]}...' -> Status: {status}, Reason: {reason}")
        
        # Update storage (we store EVERYTHING, but flags determine visibility)
        self.arguments[argument.id] = argument
        
        if status == "replaced":
            # Soft Replace: Old was hidden, New was added/shown
             if self.stance_method in ("bayesian", "agentic", "llm"):
                 if self.belief_semantics_mode in ACTIVE_SET_BELIEF_SEMANTICS_MODES:
                     change = self.belief_engine.recompute_from_active_records(
                         self.arguments.values(),
                         persist=persist_belief_state,
                     )
                     belief_updated = True
                     new_score = self.belief_engine.get_belief_score()
                     print_info(
                         f"[{self.agent_id}] Belief Recomputed: "
                         f"Delta {change:+.3f} -> Score {new_score:.3f}",
                         level=2,
                     )
                 else:
                     # Technically we should "remove" the old belief and "add" the new one for the running counter
                     # (If we were tracking exactly, but running counter is approximate anyway)
                     pass
             reason = f"Replaced (Soft): {reason}"
             
        elif status == "ignored":
            # Soft Ignore: New is hidden
            reason = f"Ignored (Soft): {reason}"
            
        elif status == "added":
            # Worldview Protection (Credence Check)
            # If the Judgement Strategy let it through as "Novel", we must check if it contradicts our worldview.
            if self.storage_mode == "worldview" and backend:
                print_debug("Worldview refutation check: START")
                is_refuted, refutation_reason = self._attempt_refutation(processed_arg, backend)
                if is_refuted:
                    status = "ignored"
                    reason = f"Refuted by Worldview: {refutation_reason}"
                    processed_arg.credence_relevant = False
                    print_debug(f"Worldview refutation triggered: {refutation_reason}")
                else:
                    print_debug("Worldview refutation check: NOT_REFUTED")
            elif self.storage_mode != "worldview":
                print_debug(f"Worldview refutation skipped: storage_mode={self.storage_mode}")
            else:
                print_debug("Worldview refutation skipped: no backend available")
            
            if status == "added" and argument.credence_relevant:
                # Log the new argument content clearly in one line with clear separation
                side_str = "AFF" if argument.polarity == "affirmative" else "NEG" if argument.polarity == "negative" else argument.polarity.upper()
                print_memory(self.agent_id, f"+ NEW [{side_str}]: \"{argument.claim[:60]}...\"", level=2)
                
                # UNIFIED UPDATE: Use BeliefEngine
                if self.stance_method in ("bayesian", "agentic", "llm"):
                    engine_backend = backend or self.backend
                    
                    # Retrieve context to inform the update (Fix for Agentic Volatility)
                    # We want to know what arguments are currently active/relevant
                    k_retrieval = min(5, len(self.arguments))
                    if k_retrieval > 0:
                        context_records = retrieve_arguments(list(self.arguments.values()), strategy="relevant", k=k_retrieval, query=argument.claim)
                    else:
                        context_records = []
                    memory_context = "\n".join([
                        f"- [{arg.polarity.upper()}] {arg.claim} (strength={arg.support_strength:.2f}, role={getattr(arg, 'source_role', 'unknown')})"
                        for arg in context_records
                    ])
                    
                    active_opponent_count = sum(
                        1
                        for arg in self.arguments.values()
                        if arg.id != argument.id
                        and getattr(arg, "credence_relevant", False)
                        and getattr(arg, "source_role", "") == "opponent"
                    )
                    if self.belief_semantics_mode in ACTIVE_SET_BELIEF_SEMANTICS_MODES:
                        change = self.belief_engine.recompute_from_active_records(
                            self.arguments.values(),
                            persist=persist_belief_state,
                        )
                    else:
                        change = self.belief_engine.update(
                            argument,
                            backend=engine_backend,
                            memory_context=memory_context,
                            persist=persist_belief_state,
                            active_opponent_count=active_opponent_count,
                        )
                    new_score = self.belief_engine.get_belief_score()
                    new_desc = self.belief_engine.get_belief_description()
                    belief_updated = True
                    
                    print_info(f"[{self.agent_id}] Belief Updated: Delta {change:+.3f} -> Score {new_score:.3f} ({new_desc})", level=2)
                    
                # Trace memory evolution
                if trace_memory:
                    self.log_memory_state()

        return status, reason, snippet, processed_arg, belief_updated

    def add_argument(self, argument: ArgumentRecord, backend=None) -> (str, str, str, ArgumentRecord):
        """
        Adds an argument to the store using the configured Evolution Strategy.
        Returns (status, reason, snippet, record).
        """
        status, reason, snippet, processed_arg, _ = self._add_argument_internal(
            argument,
            backend=backend,
            persist_belief_state=True,
            trace_memory=True,
        )
        return status, reason, snippet, processed_arg

    def add_arguments_batch(self, arguments, backend=None):
        """
        Ingest a turn's extracted arguments while deferring belief-state persistence
        and verbose memory tracing until the batch is complete.
        """
        results = []
        belief_updated = False

        for argument in arguments:
            status, reason, snippet, processed_arg, updated = self._add_argument_internal(
                argument,
                backend=backend,
                persist_belief_state=False,
                trace_memory=False,
            )
            results.append((status, reason, snippet, processed_arg))
            belief_updated = belief_updated or updated

        if belief_updated and hasattr(self.belief_engine, "save_state"):
            self.belief_engine.save_state()
            self.log_memory_state()

        return results

    def _attempt_refutation(self, argument: ArgumentRecord, backend) -> (bool, str):
        """
        Checks if the agent's existing memory contains information that effectively refutes the new argument.
        Returns (True, Reason) if refuted.
        Returns (False, "") if not refuted.
        """
        if not self.arguments:
            return False, ""

        # 0. Hard Check for Stubbornness (No LLM needed)
        if self.worldview_openness == "stubborn":
            # Calculate dominant stance
            score = sum(arg.support_strength * (1 if arg.polarity == POLARITY_AFFIRMATIVE else -1) for arg in self.arguments.values())
            current_stance = POLARITY_AFFIRMATIVE if score > 0 else POLARITY_NEGATIVE
            
            # If new argument opposes current stance, reject it immediately
            if argument.polarity != current_stance:
                return True, f"Dogmatic rejection: Argument ({argument.polarity}) opposes current worldview ({current_stance})."

        # Retrieve potentially relevant memories (RAG)
        # Use simple Vector Search (strategy="relevant")
        k_refutation = 15
        relevant_records = retrieve_arguments(
             list(self.arguments.values()), 
             strategy="relevant", 
             k=k_refutation, 
             query=argument.claim
        )
        existing_claims = [arg.claim for arg in relevant_records]
        
        print_debug(f"Refutation Check (RAG): Checking against {len(existing_claims)} most relevant beliefs.")
        
        # Adjust strictness based on openness
        if self.worldview_openness == "open_minded":
            # High Openness: Hard to refute (requires direct proof)
            strictness_instruction = """
            STRICTNESS: LOW (Open-Minded).
            - Only refute if the 'Existing Knowledge' DIRECTLY and FACTUALLY disproves the 'New Argument'.
            - Allow nuances, different perspectives, and subjective disagreements.
            - If there is any ambiguity, DO NOT refute.
            """
        elif self.worldview_openness == "stubborn":
            # Low Openness: Easy to refute (stubborn)
            strictness_instruction = """
            STRICTNESS: EXTREME (Dogmatic).
            - You are a closed-minded defender of the 'Existing Knowledge'.
            - If the 'New Argument' conflicts with the 'Existing Knowledge' in ANY way (fact, sentiment, or conclusion), you MUST refute it.
            - If the 'New Argument' comes from an opposing stance, AUTOMATICALLY refute it using the 'Existing Knowledge'.
            - Do not allow any nuance. If it's not what we already believe, it is wrong.
            """
        else:
            # Medium Openness (Default: "balanced")
            strictness_instruction = """
            STRICTNESS: MEDIUM (Balanced).
            - Refute if the 'Existing Knowledge' provides a strong, logical counter-argument.
            - Do not refute if the 'New Argument' is a valid point that can coexist with existing knowledge.
            """

        prompt = f"""
        You are a critical reasoning engine.
        Determine if the "New Argument" is effectively REFUTED by the "Existing Knowledge".
        
        New Argument: "{argument.claim}"
        
        Existing Knowledge:
        {json.dumps(existing_claims, indent=2)}
        
        {strictness_instruction}
        
        Instructions:
        1. Analyze if 'Existing Knowledge' invalidates 'New Argument' based on the STRICTNESS level above.
        2. If REFUTED: Output 'YES' followed by a brief explanation.
        3. If NOT REFUTED: Output 'NO'.
        
        Output format:
        YES: [Brief explanation]
        or
        NO
        """
        try:
            response = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a critical thinker. Output YES or NO.",
                temperature=0.0
            ).strip()
            
            if response.upper().startswith("YES"):
                reason = response[4:].strip()
                return True, reason
        except Exception as e:
            print(f"   (Memory) Refutation check failed: {e}")
        
        return False, ""

    def _should_store_argument(self, argument: ArgumentRecord, backend) -> (bool, str):
        """
        Decides if an argument is novel or important enough to store.
        Returns (True, "") or (False, Reason).
        """
        if not self.arguments:
            return True, ""
        
        # Check against ALL existing claims to prevent long-term repetition
        existing_claims = [arg.claim for arg in self.arguments.values()]

        # 1. Strict Novelty Check (Always run this first)
        skepticism_instruction = ""
        role = getattr(argument, 'source_role', 'UNKNOWN')

        if role == 'self':
            print(f"   [DEBUG] Checking Self-Argument: '{argument.claim[:30]}...' (Override Active)")
            skepticism_instruction = """
            SPECIAL INSTRUCTION (SELF-REFLECTION):
            This argument comes from the agent's own speech. Humans rarely "learn" new things just by hearing themselves talk.
            DEFAULT ACTION: Output 'NO'.
            EXCEPTION: Only output 'YES' if this argument represents a MAJOR synthesis or epiphany that connects two previously unrelated concepts.
            Refuse clear restatements, expansions, or minor logical steps. Be Ruthless.
            """
            
        # B. Continuous SATURATION Formula (Inertia)
        # Score = Memory Size * (1.0 + |Initial Stance|)
        # The more arguments you have, and the more radical you started, the harder it is to change.
        saturation_score = memory_size * (1.0 + abs(self.initial_stance))
        
        saturation_context = ""
        saturation_rules = ""
        
        if saturation_score < 15:
            # Stage 1: Formation
            saturation_context = "Your worldview is still forming. You are open to new perspectives."
            saturation_rules = "- Be strict about redundancy, but open to new concepts."
            
        elif saturation_score < 25:
             # Stage 2: Crystallization
            saturation_context = "Your worldview is taking shape. You are becoming skeptical of marginal updates."
            saturation_rules = "- Reject arguments that are minor variations of what you know.\n- Only accept clear, distinct new points."
            
        elif saturation_score < 35:
            # Stage 3: Hardening
            saturation_context = "Your worldview is well-defined. You are highly resistant to change."
            saturation_rules = "- HIGH RESISTANCE: Reject everything unless it fundamentally challenges a core assumption.\n- New arguments must be undeniably novel and critical."
            
        else:
            # Stage 4: Ossification
            saturation_context = "Your worldview is solidified. You are effectively closed to standard argumentation."
            saturation_rules = "- EXTREME RESISTANCE: Default to 'NO'.\n- Exception: Only accept 'Black Swan' arguments—facts that completely upend the debate landscape."

        # C. SELF-SKEPTICISM OVERRIDE
        # If the argument is from SELF, we ignore the saturation stage and apply MAXIMUM strictness.
        if getattr(argument, 'source_role', '') == 'self':
            saturation_context = "SELF-REFLECTION MODE. You are evaluating your own speech."
            saturation_rules = """
            - CRITICAL: Humans do not learn by repeating themselves. 
            - REJECT: Any argument that supports your current stance. 
            - REJECT: Any variation, expansion, or specific example of what you already believe.
            - ACCEPT ONLY: An argument that contradicts your previous beliefs (Self-Correction) or synthesizes opposing views.
            - DEFAULT: 'NO'.
            """

        saturation_instruction = f"""
        SATURATION STATUS (Score {saturation_score:.1f}):
        {saturation_context}
        
        STRICTNESS RULES:
        {saturation_rules}
        """

        # D. FINAL PROMPT CONSTRUCTION
        role = getattr(argument, 'source_role', 'UNKNOWN')
        
        if role == 'self':
            # NUCLEAR OPTION for Self-Arguments
            novelty_prompt = f"""
            You are a strict logical filter ensuring an agent does not learn from its own repetition.
            
            CLAIM: "{argument.claim}"
            SOURCE: The agent itself (You).
            
            TASK: Decide if this argument is a MAJOR BREAKTHROUGH.
            
            RULES:
            1. If this argument SUPPORTS your current stance -> OUTPUT 'NO'. (Self-reinforcement is banned).
            2. If this argument is a variation of what you already believe -> OUTPUT 'NO'.
            3. OUTPUT 'YES' ONLY if it is a SELF-CORRECTION (admitting you were wrong) or a SYNTHESIS of opposing views.
            
            DEFAULT ANSWER: NO.
            
            Response Format: explanation, then 'YES' or 'NO'.
            """
        else:
            novelty_prompt = f"""
            You are a hostile memory filter. Your Default Answer is NO.
            
            Task: check if "New Argument" is worth adding to "Existing Claims".
            
            {skepticism_instruction}
            
            Rules:
            1. REDUNDANCY: If the New Argument is semantically similar to ANY existing claim, output 'NO'.
            2. SPECIFICITY: If the New Argument is just a specific example of a general claim you already have, output 'NO'.
            3. SATURATION: Apply the following filter strictly:
            {saturation_instruction}
            
            New Argument: "{argument.claim}"
            
            Existing Claims:
            {json.dumps(existing_claims, indent=2)}
            
            Is the New Argument NOVEL and IMPORTANT enough to store?
            Response Format: explanation (1 sentence), then 'YES' or 'NO'.
            Output ONLY 'YES' or 'NO'.
            """
        try:
            is_novel = backend.chat(
                messages=[{"role": "user", "content": novelty_prompt}],
                system_prompt="You are a strict filter. Output ONLY YES or NO.",
                temperature=0.0
            ).strip().upper()
            
            if "NO" in is_novel:
                return False, "Duplicate or semantically equivalent to existing memory"
        except Exception as e:
            print(f"   (Memory) Novelty check failed: {e}")
            # If check fails, assume it's novel to be safe? Or reject? 
            # Let's assume novel to avoid losing data on error, but log it.
            pass

        # Note: We removed the "Worldview Fit Check" because the user wants to accept arguments 
        # that cannot be refuted, even if they don't fit the worldview.

        return True, ""

    def _check_contradiction_and_replace(self, new_argument: ArgumentRecord, backend) -> (str, str, str):
        """
        Checks if the new argument contradicts any existing argument.
        If it does, and the new argument is deemed 'better' or simply accepted as the new truth,
        it returns the ID of the argument to replace.
        
        Returns: (replaced_arg_id, reason, replaced_claim) or (None, "", "")
        """
        if not self.arguments:
            return None, "", ""

        existing_claims = [{"id": arg.id, "claim": arg.claim, "polarity": arg.polarity} for arg in self.arguments.values()]

        prompt = f"""
        You are an agentic reasoning engine.
        Your goal is to evolve the agent's knowledge by replacing outdated, weaker, or less relevant beliefs with better ones.
        
        New Argument: "{new_argument.claim}" ({new_argument.polarity})
        
        Existing Knowledge:
        {json.dumps(existing_claims, indent=2)}
        
        Instructions:
        1. Compare the 'New Argument' against the 'Existing Knowledge'.
        2. Identify if any existing argument should be REPLACED by the new one.
        3. Replacement Criteria (any of the following):
           - CONTRADICTION: New argument disproves or overrides the old one.
           - RELEVANCE: New argument is significantly more relevant to the core topic than the old one.
           - SPECIFICITY: New argument provides specific evidence/examples where the old one was vague.
           - QUALITY: New argument is simply better articulated, more persuasive, or more comprehensive.
        4. If the New Argument is just a different aspect of the same truth and both are valuable, DO NOT replace.
        
        Output format:
        REPLACE: [ID of the existing argument to delete]
        REASON: [Brief explanation citing Contradiction, Relevance, Specificity, or Quality]
        
        If no replacement is needed, output:
        NO_CONTRADICTION
        """
        
        try:
            response = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are an agentic engine. Output REPLACE or NO_CONTRADICTION.",
                temperature=0.0
            ).strip()
            
            if "REPLACE:" in response:
                lines = response.split('\n')
                replaced_id = None
                reason = "Contradiction found"
                
                for line in lines:
                    if line.startswith("REPLACE:"):
                        replaced_id = line.replace("REPLACE:", "").strip()
                    if line.startswith("REASON:"):
                        reason = line.replace("REASON:", "").strip()
                
                if replaced_id:
                    # Verify ID exists and get claim
                    for arg in self.arguments.values():
                        if arg.id == replaced_id:
                            return replaced_id, reason, arg.claim
                        
        except Exception as e:
            print(f"   (Memory) Agentic check failed: {e}")
            
        return None, "", ""
            
    def _check_synthesis(self, new_argument: ArgumentRecord, backend) -> (str, str, str):
        """
        Checks if the new argument can be synthesized with an existing argument to form a higher-order belief.
        Returns: (subsumed_arg_id, synthesis_claim, reason) or (None, "", "")
        """
        if not self.arguments:
            return None, "", ""

        existing_claims = [{"id": arg.id, "claim": arg.claim, "polarity": arg.polarity} for arg in self.arguments.values()]

        prompt = f"""
        You are a dialectical reasoning engine.
        Your goal is to synthesize opposing views into a higher-order truth (Thesis + Antithesis = Synthesis).
        
        New Argument (Antithesis): "{new_argument.claim}" ({new_argument.polarity})
        
        Existing Knowledge (Theses):
        {json.dumps(existing_claims, indent=2)}
        
        Instructions:
        1. Look for an existing argument that conflicts with or opposes the New Argument.
        2. Can you construct a SYNTHESIS claim that incorporates BOTH truths?
           - The synthesis must be nuanced, conditional, or higher-level.
           - It should resolve the contradiction.
        3. If yes, output SYNTHESIS.
        
        Output format:
        SYNTHESIS: [The new synthesized claim]
        SUBSUMES: [ID of the existing argument that is now part of the synthesis]
        REASON: [Brief explanation of the synthesis]
        
        If no synthesis is possible, output:
        NO_SYNTHESIS
        """
        
        try:
            response = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a dialectical engine. Output SYNTHESIS or NO_SYNTHESIS.",
                temperature=0.0
            ).strip()
            
            if "SYNTHESIS:" in response:
                lines = response.split('\n')
                subsumed_id = None
                synthesis_claim = ""
                reason = "Synthesis found"
                
                for line in lines:
                    if line.startswith("SYNTHESIS:"):
                        synthesis_claim = line.replace("SYNTHESIS:", "").strip()
                    if line.startswith("SUBSUMES:"):
                        subsumed_id = line.replace("SUBSUMES:", "").strip()
                    if line.startswith("REASON:"):
                        reason = line.replace("REASON:", "").strip()
                
                if subsumed_id and synthesis_claim:
                    return subsumed_id, synthesis_claim, reason
                        
        except Exception as e:
            print(f"   (Memory) Synthesis check failed: {e}")
            
        return None, "", ""

    def get_bayesian_probability(self) -> float:
        """
        Returns the raw Bayesian probability (0.0 to 1.0) using only credence_relevant arguments.
        Returns 0.5 if not in Bayesian mode.
        """
        if self.stance_method != "bayesian":
            return 0.5
        
        # Delegate to engine
        stance = self.belief_engine.get_belief_score()
        return (stance + 1.0) / 2.0

    def get_stance(self, verbose: bool = True, **kwargs) -> float:
        """
        Computes the current stance using only credence_relevant arguments.
        """
        # Filter to only credence_relevant arguments
        credence_relevant_args = [
            arg for arg in self.arguments.values()
            if arg.credence_relevant
        ]
        
        # Calculate statistics for verbose output
        pro_args = [arg for arg in credence_relevant_args if arg.polarity == POLARITY_AFFIRMATIVE]
        con_args = [arg for arg in credence_relevant_args if arg.polarity == POLARITY_NEGATIVE]
        
        pro_count = len(pro_args)
        con_count = len(con_args)
        
        pro_mean_strength = sum(arg.support_strength for arg in pro_args) / pro_count if pro_count > 0 else 0.0
        con_mean_strength = sum(arg.support_strength for arg in con_args) / con_count if con_count > 0 else 0.0
        
        # Compute stance (Standardized on BeliefEngine)
        # Compute stance based on method
        if self.stance_method in ("bayesian", "agentic", "llm"):
            stance = self.belief_engine.get_belief_score()
            if verbose:
                 engine_label = "AgenticEngine" if self.stance_method in ("agentic", "llm") else "BayesianEngine"
                 print_info(f"[{self.agent_id}] Stance ({engine_label}): {stance:+.4f}", level=2)
        else:
            # Linear / Aggregate Method (Used for 'linear' and 'blind')
            # Calculate total mass
            pro_mass = sum(arg.support_strength for arg in pro_args)
            con_mass = sum(arg.support_strength for arg in con_args)
            total_mass = pro_mass + con_mass
            
            if total_mass > 0:
                stance = (pro_mass - con_mass) / total_mass
            else:
                stance = 0.0
                
            if verbose:
                method_name = "Blind" if self.stance_method == "blind" else "Linear"
                print_info(f"[{self.agent_id}] Stance ({method_name}): {stance:+.4f} (Pro Mass: {pro_mass:.1f}, Con Mass: {con_mass:.1f})", level=2)
        
        return stance


    def retrieve(self, k: int = 3, **kwargs) -> List[ArgumentRecord]:
        """
        Retrieves arguments based on the configured strategy.
        kwargs can include 'query' and 'backend' for 'relevant' strategy.
        """
        # If weighted, we need to pass the current stance
        if self.retrieval_strategy == "weighted":
            kwargs["current_stance"] = self.get_stance()
            
        return retrieve_arguments(list(self.arguments.values()), strategy=self.retrieval_strategy, k=k, **kwargs)


    def get_all_arguments(self) -> List[Dict]:
        """
        Returns all arguments as a list of dictionaries (for logging/serialization).
        """
        return [arg.to_dict() for arg in self.arguments.values()]


    def log_memory_state(self):
        """Prints a structured summary of the current memory state (verbose only)."""
        if get_verbosity() < 2:
            return
            
        if not self.arguments:
            return

        total = len(self.arguments)
        active_args = [arg for arg in self.arguments.values() if arg.credence_relevant]
        active_count = len(active_args)
        
        aff_active = len([arg for arg in active_args if arg.polarity == POLARITY_AFFIRMATIVE])
        neg_active = len([arg for arg in active_args if arg.polarity == POLARITY_NEGATIVE])
        
        aff_strength = sum(arg.support_strength for arg in active_args if arg.polarity == POLARITY_AFFIRMATIVE) / max(1, aff_active)
        neg_strength = sum(arg.support_strength for arg in active_args if arg.polarity == POLARITY_NEGATIVE) / max(1, neg_active)

        print_info(f"[{self.agent_id}] Memory: {active_count}/{total} active | AFF: {aff_active} (μ={aff_strength:.2f}) | NEG: {neg_active} (μ={neg_strength:.2f})", level=2)
