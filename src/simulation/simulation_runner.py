from typing import List, Dict, Any, Optional
import csv
import json
import logging
from datetime import datetime
from .agent_interface import DebatingAgent
from src.agents.extractor_agent import ArgumentExtractionAgent
from src.memory.argument_record import ArgumentRecord
from src.utils.memory_diff import generate_memory_diff_md
import os

SPEAKER_ORDERS = ("pro-first", "con-first")


class SimulationRunner:
    """
    A generic runner for multi-agent debates.
    Supports N agents, alternating turns, and tracking stance evolution for all participants.
    """
    def __init__(
        self,
        agents: List[DebatingAgent],
        rounds: int = 10,
        extractor: Optional[ArgumentExtractionAgent] = None,
        evaluator: Any = None,
        output_dir: Optional[str] = None,
        initial_transcript: List[Dict] = None,
        start_round: int = 1,
        skip_subject_extraction: bool = False,
        live_plot: bool = False,
        live_json: bool = True,
        artifact_mode: str = "full",
        evaluator_strict: bool = False,
        extraction_strict: bool = False,
        memory_strict: bool = False,
        speaker_order: str = "pro-first",
    ):
        if speaker_order not in SPEAKER_ORDERS:
            raise ValueError(
                f"Unknown speaker_order {speaker_order!r}; expected one of "
                f"{SPEAKER_ORDERS}."
            )
        if speaker_order == "con-first" and len(agents) != 2:
            raise ValueError(
                "speaker_order='con-first' is defined only for two-agent "
                "simulations."
            )
        self.agents = agents
        self.rounds = rounds
        self.extractor = extractor
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.transcript: List[Dict[str, str]] = initial_transcript if initial_transcript else []
        self.start_round = start_round
        self.stance_history: List[Dict[str, float]] = []
        self.argument_updates: List[Dict[str, Any]] = []
        self.skip_subject_extraction = skip_subject_extraction
        self.live_plot = live_plot
        self.live_json = live_json
        self.artifact_mode = artifact_mode
        self.evaluator_strict = bool(evaluator_strict)
        self.extraction_strict = bool(extraction_strict)
        self.memory_strict = bool(memory_strict)
        self.speaker_order = speaker_order
        self.logger = logging.getLogger(__name__)

    def _turn_order(self):
        """Return indexed agents in the configured within-round order."""
        indexed_agents = list(enumerate(self.agents))
        if self.speaker_order == "con-first":
            return [indexed_agents[1], indexed_agents[0]]
        return indexed_agents

    def _prepare_opponent_evidence_receivers(self, speaker, round_index):
        """Prepare receiver memories after successful extraction, never sender."""
        prepared = []
        try:
            for receiver in self.agents:
                if receiver is speaker:
                    continue
                memory = getattr(receiver, "memory", None)
                prepare = getattr(
                    memory,
                    "prepare_for_opponent_evidence_turn",
                    None,
                )
                if not callable(prepare):
                    continue
                finalize = getattr(
                    memory,
                    "finalize_opponent_evidence_turn",
                    None,
                )
                if not callable(finalize):
                    raise RuntimeError(
                        f"{receiver.name} exposes a horizon prepare hook "
                        "without the matching finalize hook."
                    )
                event_index = prepare(
                    round_index=round_index,
                    sender_name=speaker.name,
                )
                prepared.append((receiver, event_index))
        except Exception as exc:
            self._abort_opponent_evidence_receivers(prepared, exc)
            raise
        return prepared

    @staticmethod
    def _abort_opponent_evidence_receivers(prepared, error):
        for receiver, event_index in prepared:
            abort = getattr(
                receiver.memory,
                "abort_opponent_evidence_turn",
                None,
            )
            if callable(abort):
                abort(event_index, reason=str(error))

    @staticmethod
    def _agent_audit_bundle(
        agents,
        method_name,
        *,
        additive_keys=(),
        maximum_keys=(),
    ):
        agent_audits = {}
        totals = {key: 0 for key in additive_keys}
        maxima = {key: 0 for key in maximum_keys}
        for index, agent in enumerate(agents):
            audit_method = getattr(agent, method_name, None)
            if not callable(audit_method):
                continue
            summary = audit_method()
            audit_name = getattr(
                agent,
                "name",
                getattr(agent, "agent_id", f"agent_{index}"),
            )
            agent_audits[str(audit_name)] = summary
            for key in additive_keys:
                totals[key] += int(summary.get(key, 0))
            for key in maximum_keys:
                maxima[key] = max(
                    float(maxima[key]),
                    float(summary.get(key, 0)),
                )
        if not agent_audits:
            return None
        return {
            "agents": agent_audits,
            "totals": totals,
            "maxima": maxima,
        }

    def run(self) -> Dict[str, Any]:
        """
        Executes the simulation loop.
        Returns a dictionary containing the transcript and stance history.
        """
        self.logger.info(f"Starting simulation with {len(self.agents)} agents for {self.rounds} rounds.")
        
        # 0. Initial Stance Capture & Memory Snapshot
        # Snapshot the initial argument set for diffing later.
        initial_argument_ids: Dict[str, set] = {}
        initial_stances: Dict[str, float] = {}
        
        for agent in self.agents:
            initial_stances[agent.name] = agent.get_current_stance()
            if getattr(agent, "memory", None) is not None:
                initial_argument_ids[agent.name] = set(agent.memory.arguments.keys())
            else:
                initial_argument_ids[agent.name] = set()
        self._record_stances(round_index=0)
        
        try:
            # 1. Main Loop
            for r in range(self.start_round, self.start_round + self.rounds):
                self.logger.info(f"\n{'='*50}\n=== Round {r} ===\n{'='*50}\n")
                
                # Keep original indices so chat roles remain attached to agent
                # identity even when the two-agent sensitivity reverses order.
                turn_order = self._turn_order()
                
                for original_idx, agent in turn_order:
                    # Generate reply
                    response_text, metadata = agent.generate_reply(self.transcript)
                    
                    role = "assistant" if original_idx == 0 else "user"
                    
                    message = {
                        "role": role,
                        "content": response_text,
                        "agent_name": agent.name
                    }
                    self.transcript.append(message)
                    self.logger.info(f"\n[{agent.name}]:\n\"{response_text}\"\n")

                    # --- 2. Extract and Distribute Arguments ---
                    if self.extractor:
                        prepared_receivers = []
                        try:
                            # Use 'extract' method, passing expected source_role="opponent" by default
                            # OPTIMIZATION: If configured, skip extraction for the Subject (Agent 0)
                            # This saves time when we don't need self-arguments.
                            
                            should_extract = True
                            if getattr(self, "skip_subject_extraction", False) and agent.name == "Subject":
                                should_extract = False
                                
                            if should_extract:
                                extracted_args = self.extractor.extract(
                                    text=response_text, 
                                    source_role="opponent"
                                )
                                prepared_receivers = (
                                    self._prepare_opponent_evidence_receivers(
                                        agent,
                                        r,
                                    )
                                )
                            else:
                                extracted_args = []
                                prepared_receivers = []

                            receiver_batch_audits = {
                                receiver.name: {
                                    "accepted_records": 0,
                                    "status_counts": {},
                                }
                                for receiver, _event_index in prepared_receivers
                            }
                            
                            if extracted_args:
                                self.logger.info(f"   (System) Extracted {len(extracted_args)} arguments from {agent.name}.")
                                
                                # Distribute to ALL agents
                                for other_agent in self.agents:
                                    # Create specific list for this agent (handling self-reinforcement)
                                    agent_specific_args = []
                                    is_self = (other_agent == agent)
                                    
                                    for arg in extracted_args:
                                        # Use source_role="self" if we are reinforcing our own argument
                                        role_for_storage = "self" if is_self else "opponent"
                                        
                                        new_arg = ArgumentRecord(
                                            topic_id=arg.topic_id,
                                            claim=arg.claim,
                                            polarity=arg.polarity,
                                            source_role=role_for_storage,
                                            support_strength=arg.support_strength,
                                            support_strength_llm=(
                                                arg.support_strength_llm
                                            ),
                                            support_strength_classifier=(
                                                arg.support_strength_classifier
                                            ),
                                            evidence_snippets=arg.evidence_snippets,
                                            category_tags=arg.category_tags,
                                            timestamp=arg.timestamp
                                        )
                                        agent_specific_args.append(new_arg)

                                    # Add to memory
                                    if hasattr(other_agent, "memory") and hasattr(other_agent.memory, "add_argument"):
                                        added_count = 0
                                        backend = getattr(other_agent, "backend", None)

                                        try:
                                            if hasattr(other_agent.memory, "add_arguments_batch"):
                                                batch_results = other_agent.memory.add_arguments_batch(agent_specific_args, backend)
                                            else:
                                                batch_results = [
                                                    other_agent.memory.add_argument(arg, backend)
                                                    for arg in agent_specific_args
                                                ]

                                            for arg, result in zip(agent_specific_args, batch_results):
                                                status, reason, snippet, processed_arg = result
                                                if other_agent.name in receiver_batch_audits:
                                                    status_counts = receiver_batch_audits[
                                                        other_agent.name
                                                    ]["status_counts"]
                                                    status_counts[status] = (
                                                        status_counts.get(status, 0) + 1
                                                    )

                                                if status in ["added", "replaced", "synthesized", "ignored"]:
                                                    if status != "ignored":
                                                        added_count += 1

                                                    self.argument_updates.append({
                                                        "round": r,
                                                        "receiver": other_agent.name,
                                                        "sender": agent.name,
                                                        "status": status,
                                                        "claim": processed_arg.claim if processed_arg else arg.claim,
                                                        "reason": reason,
                                                        "strength": processed_arg.support_strength if processed_arg else 0.0,
                                                        "timestamp": datetime.now().isoformat()
                                                    })
                                        except Exception as e:
                                            self.logger.warning(f"Failed to add arguments to {other_agent.name}: {e}")
                                            if self.memory_strict:
                                                raise
                                        
                                        if added_count > 0:
                                            self.logger.info(f"      + {other_agent.name} internalized {added_count} new arguments.")
                                        if other_agent.name in receiver_batch_audits:
                                            receiver_batch_audits[
                                                other_agent.name
                                            ]["accepted_records"] = added_count

                            for receiver, event_index in prepared_receivers:
                                batch_audit = receiver_batch_audits[receiver.name]
                                receiver.memory.finalize_opponent_evidence_turn(
                                    event_index,
                                    extracted_records=len(extracted_args),
                                    accepted_records=batch_audit[
                                        "accepted_records"
                                    ],
                                    status_counts=batch_audit["status_counts"],
                                )
                                            
                        except Exception as e:
                            self._abort_opponent_evidence_receivers(
                                prepared_receivers,
                                e,
                            )
                            self.logger.error(f"Argument extraction error: {e}")
                            if self.extraction_strict or self.memory_strict:
                                raise

                    for observer in self.agents:
                        observe_message = getattr(observer, "observe_message", None)
                        if not observe_message:
                            continue
                        try:
                            observe_message(message, self.transcript)
                        except Exception as e:
                            self.logger.warning(
                                "Observation hook failed for %s after %s spoke: %s",
                                getattr(observer, "name", "agent"),
                                agent.name,
                                e,
                            )
                
                # Record stances after everyone has spoken
                self._record_stances(round_index=r)
            
            print(f"Simulation completed: {len(self.transcript)} messages exchanged.")

        finally:
            # Generate Memory Reports
            try:
                # Use explicit output dir if available, else fallback to current dir
                log_dir = self.output_dir if self.output_dir else "."
                
                # Check for logger handler fallback ONLY if explicit dir is missing (optional)
                if not self.output_dir:
                     for handler in self.logger.handlers:
                        if hasattr(handler, 'baseFilename'):
                            log_dir = os.path.dirname(handler.baseFilename)
                            break

                for agent in self.agents:
                    if getattr(agent, "memory", None) is not None:
                        final_stance = agent.get_current_stance()
                        initial_stance = initial_stances.get(agent.name, 0.0)
                        initial_ids = initial_argument_ids.get(agent.name, set())
                        
                        top_val = "Unknown"
                        if agent.memory.arguments:
                             first_arg = list(agent.memory.arguments.values())[0]
                             # Handle both Object and Dict
                             if isinstance(first_arg, dict):
                                 top_val = first_arg.get('topic_id', 'Unknown')
                             else:
                                 top_val = getattr(first_arg, 'topic_id', 'Unknown')
                        
                        report_md = generate_memory_diff_md(
                            agent_name=agent.name,
                            final_memory=agent.memory,
                            initial_arg_ids=initial_ids,
                            initial_stance=initial_stance,
                            final_stance=final_stance,
                            topic=top_val
                        )
                        
                        filename = f"memory_diff_{agent.name}.md"
                        if self.artifact_mode == "compact":
                            filename = os.path.join("memory_reports", filename)
                        filepath = os.path.join(log_dir, filename)
                        os.makedirs(os.path.dirname(filepath), exist_ok=True)
                        with open(filepath, "w") as f:
                            f.write(report_md)
                        
            except Exception as e:
                self.logger.error(f"Failed to generate memory diff reports: {e}")

        result = {
            "transcript": self.transcript,
            "stance_history": self.stance_history,
            "argument_updates": self.argument_updates,
            "metadata": {
                "simulation": {
                    "speaker_order": self.speaker_order,
                    "turn_order": [
                        agent.name for _, agent in self._turn_order()
                    ],
                }
            },
        }
        audit_summary = getattr(self.extractor, "audit_summary", None)
        if callable(audit_summary):
            result.setdefault("metadata", {})[
                "argument_extraction"
            ] = audit_summary()
        baseline_agent_audits = {}
        baseline_audit_totals = {}
        for agent in self.agents:
            agent_audit_summary = getattr(
                agent,
                "self_update_audit_summary",
                None,
            )
            if not callable(agent_audit_summary):
                continue
            summary = agent_audit_summary()
            baseline_agent_audits[agent.name] = summary
            for key, value in summary.items():
                if isinstance(value, int) and not isinstance(value, bool):
                    baseline_audit_totals[key] = (
                        baseline_audit_totals.get(key, 0) + value
                    )
        if baseline_agent_audits:
            result.setdefault("metadata", {})[
                "self_update_baseline"
            ] = {
                "agents": baseline_agent_audits,
                "totals": baseline_audit_totals,
            }
        horizon_agents = [
            agent.memory
            for agent in self.agents
            if getattr(agent, "memory", None) is not None
        ]
        audit_specs = (
            (
                "response_retrieval",
                self.agents,
                "response_retrieval_audit_summary",
                (
                    "reply_calls",
                    "retrieval_calls",
                    "retrieval_skipped_calls",
                    "records_returned_total",
                    "memory_context_tokens_total",
                    "memory_context_budget_saturated_calls",
                ),
                (
                    "records_returned_max",
                    "memory_context_tokens_max",
                ),
            ),
            (
                "dynamic_evidence_horizon",
                horizon_agents,
                "evidence_horizon_audit_summary",
                (
                    "prepare_calls",
                    "completed_turns",
                    "aborted_turns",
                    "open_turns",
                    "reset_calls",
                    "evicted_records",
                    "seed_state_restored_records",
                    "extracted_records",
                    "accepted_records",
                ),
                (
                    "prior_reconstruction_error_max",
                    "active_state_recomputation_error_max",
                    "post_batch_dynamic_records_max",
                ),
            ),
            (
                "deduplication",
                horizon_agents,
                "deduplication_audit_summary",
                ("no_dedup_id_collisions_preserved",),
                (),
            ),
        )
        for key, agents, method, additive, maxima in audit_specs:
            bundle = self._agent_audit_bundle(
                agents,
                method,
                additive_keys=additive,
                maximum_keys=maxima,
            )
            if bundle:
                result.setdefault("metadata", {})[key] = bundle
        return result

    def _record_stances(self, round_index: int):
        """Records the current stance of all agents and displays a summary."""
        snapshot = {"round": round_index}
        stance_changes = []
        
        # Helper to format transcript for evaluator
        # (Fetches only the latest message per agent)
        
        for agent in self.agents:
            # 1. Internal Stance
            try:
                current_stance = agent.get_current_stance()
                snapshot[agent.name] = current_stance
                
                # Calculate change from previous round
                if self.stance_history:
                    # Find previous internal stance
                    prev_stance = self.stance_history[-1].get(agent.name, 0.0)
                    change = current_stance - prev_stance
                    stance_changes.append((agent.name, current_stance, change))
                else:
                    stance_changes.append((agent.name, current_stance, 0.0))
            except Exception as e:
                self.logger.warning(f"Failed to get stance for agent {agent.name}: {e}")
                snapshot[agent.name] = 0.0
                stance_changes.append((agent.name, 0.0, 0.0))

            # 2. Evaluated Stance (External Judge)
            if self.evaluator: 
                # DEBUG: Check why evaluation might be skipped
                # self.logger.info(f"DEBUG: Checking evaluation for {agent.name} (Type: {getattr(agent, 'type', 'None')})")
                
                if getattr(agent, 'type', '') in ('subject', 'vanilla', 'moltbot', 'self_update'):
                    try:
                        # We pass the full transcript + specific topic from agent config if available
                        topic = self.extractor.topic_id if self.extractor else "The Simulation Topic"
                        
                        # We only evaluate if there is a transcript (round > 0)
                        if round_index > 0:
                            # Find the last message from this agent
                            last_message = ""
                            for msg in reversed(self.transcript):
                                if msg.get('agent_name') == agent.name:
                                    last_message = msg['content']
                                    break
                            
                            # We only evaluate if we found a message
                            if last_message:
                                evaluator_kwargs = {
                                    "transcript": last_message,
                                    "memory_snapshot": "",
                                    "topic": topic,
                                    "agent_name": agent.name,
                                }
                                if self.evaluator_strict:
                                    evaluator_kwargs["strict"] = True
                                eval_result = self.evaluator.evaluate_stance(
                                    **evaluator_kwargs
                                )
                                eval_score = eval_result.get("stance_score", 0.0)
                                conf_score = eval_result.get("confidence_score", 0.0)
                                snapshot[f"{agent.name}_evaluated"] = eval_score
                                snapshot[f"{agent.name}_confidence"] = conf_score
                                self.logger.info(f"   (Judge) {agent.name} Evaluated Stance: {eval_score:.2f} (Conf: {conf_score:.2f})")
                            else:
                                self.logger.warning(f"DEBUG: No message found for {agent.name} in round {round_index}")
                        else:
                            # Initial round: Assume evaluated stance matches internal or seed?
                            snapshot[f"{agent.name}_evaluated"] = snapshot[agent.name]
                            
                    except Exception as e:
                        self.logger.error(f"Evaluation failed for {agent.name}: {e}")
                        if self.evaluator_strict:
                            raise
                        snapshot[f"{agent.name}_evaluated"] = 0.0
                else:
                    self.logger.warning(f"DEBUG: Agent {agent.name} type '{getattr(agent, 'type', '')}' not in whitelist.")
            else:
                pass # self.logger.warning("DEBUG: self.evaluator is None")

        self.stance_history.append(snapshot)
        if self.live_plot:
            self._write_live_stance_artifacts()
        
        # Display stance summary
        self.logger.info(f"\n   === Stance Summary (Round {round_index}) ===")
        for name, stance, change in stance_changes:
            change_str = f"{change:+.4f}" if round_index > 0 else "(initial)"
            
            # Check for evaluated score in snapshot
            eval_key = f"{name}_evaluated"
            if eval_key in snapshot:
                eval_score = snapshot[eval_key]
                self.logger.info(f"   {name}: Internal {stance:+.4f} [{change_str}] | Judge {eval_score:+.4f}")
            else:
                self.logger.info(f"   {name}: {stance:+.4f} [{change_str}]")

    def _write_live_stance_artifacts(self):
        """Writes live stance CSV/JSON/PNG artifacts after each recorded round."""
        if not self.output_dir or not self.stance_history:
            return

        try:
            os.makedirs(self.output_dir, exist_ok=True)
            agent_names = [agent.name for agent in self.agents]

            csv_path = os.path.join(self.output_dir, "live_stance_history.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["round", *agent_names])
                writer.writeheader()
                for snapshot in self.stance_history:
                    writer.writerow(
                        {
                            "round": snapshot.get("round"),
                            **{name: snapshot.get(name, "") for name in agent_names},
                        }
                    )

            if self.live_json:
                json_path = os.path.join(self.output_dir, "live_stance_history.json")
                with open(json_path, "w") as f:
                    json.dump(self.stance_history, f, indent=2)

            self._write_live_stance_plot(agent_names)
        except Exception as e:
            self.logger.warning(f"Failed to write live stance artifacts: {e}")

    def _write_live_stance_plot(self, agent_names: List[str]):
        """Writes a compact PNG trajectory plot for live monitoring."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            self.logger.warning(f"Live plotting unavailable: {e}")
            return

        rounds = [snapshot.get("round") for snapshot in self.stance_history]
        colors = ["#2F6F4E", "#A13D3D", "#577590", "#A9793B"]

        fig, ax = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
        for idx, name in enumerate(agent_names):
            values = [snapshot.get(name) for snapshot in self.stance_history]
            ax.plot(
                rounds,
                values,
                marker="o",
                linewidth=2.2,
                markersize=4,
                color=colors[idx % len(colors)],
                label=name,
            )
            if values:
                ax.annotate(
                    f"{values[-1]:+.2f}",
                    xy=(rounds[-1], values[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=9,
                    color=colors[idx % len(colors)],
                    fontweight="bold",
                )

        ax.axhline(0, color="#111827", linewidth=0.8, alpha=0.6)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Round")
        ax.set_ylabel("Internal stance")
        ax.set_title("Live Stance Trajectory", fontweight="bold")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=max(1, len(agent_names)))
        fig.savefig(os.path.join(self.output_dir, "live_stance_trajectory.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)
