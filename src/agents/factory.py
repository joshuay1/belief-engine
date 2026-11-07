"""
Unified agent creation factory.

Creates agents based on config dictionaries, supporting all agent types:
  - subject: Memory-backed debating agent (episodic/worldview)
  - counter: Opposing debating agent
  - vanilla: No-memory agent (conversation history only)
  - self_update: Prompt-only self-reported stance baseline, optionally with memory/RAG
  - moltbot: Moltbot agent with belief engine
  - deterministic: Scripted agent for controlled experiments
"""

import logging
import os

logger = logging.getLogger(__name__)


def create_agent(agent_config: dict, global_config: dict, backend, log_dir: str = None):
    """
    Create an agent instance based on its config.

    Args:
        agent_config: Per-agent config dict (type, name, persona, memory overrides, etc.)
        global_config: Full experiment config (backend, memory, debate sections)
        backend: LLM backend instance (from backend factory)
        log_dir: Optional log directory path (used by Moltbot for persistent storage)

    Returns:
        An agent instance (SubjectAgent, CounterAgent, VanillaSubjectAgent, etc.)

    Raises:
        ValueError: If agent type is unknown.
    """
    agent_type = agent_config.get("type", "subject")
    name = agent_config.get("name", "Unknown Agent")
    persona = agent_config.get("persona", "You are a helpful assistant.")

    if agent_type == "subject":
        return _create_subject_agent(agent_config, global_config, backend, name, persona, log_dir)

    elif agent_type == "counter":
        return _create_counter_agent(agent_config, global_config, backend)

    elif agent_type == "vanilla":
        return _create_vanilla_agent(agent_config, global_config, backend, name, persona)

    elif agent_type == "self_update":
        return _create_self_updating_agent(agent_config, global_config, backend, name, persona, log_dir)

    elif agent_type == "moltbot":
        return _create_moltbot_agent(agent_config, global_config, backend, name, persona, log_dir)

    elif "deterministic" in agent_type:
        return _create_deterministic_agent(agent_config, global_config, name)

    else:
        raise ValueError(f"Unknown agent type: '{agent_type}'. Supported: subject, counter, vanilla, self_update, moltbot, deterministic")


# --- Private helpers ---

def _create_subject_agent(agent_config, global_config, backend, name, persona, log_dir=None):
    """Create a memory-backed subject agent."""
    from src.memory.memory_store import MemoryStore
    from src.agents.subject_agent import SubjectAgent
    from src.simulation.belief_engine import (
        BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1,
        BELIEF_SEMANTICS_ACTIVE_SEED_V1,
        BELIEF_SEMANTICS_INCREMENTAL_V1,
    )
    from src.simulation.seeding_helper import seed_review_artifactory

    memory_conf = global_config.get("memory", {}).copy()
    if "memory" in agent_config:
        memory_conf.update(agent_config["memory"])
    generation_conf = global_config.get("generation", {})
    temperature = agent_config.get("temperature", generation_conf.get("temperature", 0.7))
    response_shell = agent_config.get(
        "response_shell",
        generation_conf.get("response_shell", "legacy"),
    )
    response_context_messages = agent_config.get(
        "response_context_messages",
        generation_conf.get("response_context_messages", 4),
    )
    response_word_limit = agent_config.get(
        "response_word_limit",
        generation_conf.get("response_word_limit", 120),
    )
    retrieval_k = agent_config.get(
        "retrieval_k",
        memory_conf.get("retrieval_k", 10),
    )
    response_retrieval_enabled = agent_config.get(
        "response_retrieval_enabled",
        memory_conf.get("response_retrieval_enabled", True),
    )
    retrieved_evidence_token_budget = agent_config.get(
        "retrieved_evidence_token_budget",
        memory_conf.get("retrieved_evidence_token_budget"),
    )
    retrieved_evidence_encoding = agent_config.get(
        "retrieved_evidence_encoding",
        memory_conf.get("retrieved_evidence_encoding", "cl100k_base"),
    )
    belief_semantics_mode = memory_conf.get(
        "belief_semantics_mode",
        BELIEF_SEMANTICS_INCREMENTAL_V1,
    )

    storage_mode = memory_conf.get("storage_mode", "episodic")
    common_kwargs = {
        "storage_mode": storage_mode,
        "retrieval_strategy": memory_conf.get("retrieval_strategy", "weighted"),
        "worldview_openness": memory_conf.get("worldview_openness", "balanced"),
        "stance_method": memory_conf.get("stance_method", "bayesian"),
        "update_sensitivity": memory_conf.get("update_sensitivity", 0.3),
        "anchor_sensitivity": memory_conf.get("anchor_sensitivity", 0.1),
        "confirmation_bias": memory_conf.get("confirmation_bias", 0.0),
        "allow_self_arguments": memory_conf.get("allow_self_arguments", True),
        "belief_engine_type": memory_conf.get("belief_engine_type", "bayesian"),
        "belief_semantics_mode": belief_semantics_mode,
        "backend": backend,
        "agent_id": name,
        "initial_stance": agent_config.get("target_initial_stance", 0.0),
        "argument_similarity_threshold": memory_conf.get("argument_similarity_threshold", 0.7),
        "self_similarity_threshold": memory_conf.get("self_similarity_threshold", 0.95),
        "cross_polarity_dedup": memory_conf.get("cross_polarity_dedup", False),
        "deduplication_enabled": memory_conf.get(
            "deduplication_enabled",
            True,
        ),
        "dynamic_evidence_horizon": memory_conf.get(
            "dynamic_evidence_horizon",
            "persistent",
        ),
    }
    if log_dir:
        common_kwargs["memory_dir"] = os.path.join(log_dir, "memory", name)

    if storage_mode == "agentic":
        from src.memory.agentic_memory import AgenticMemoryStore
        memory = AgenticMemoryStore(
            llm_backend=global_config["backend"].get("type", "openai"),
            llm_model=global_config["backend"].get("model", "gpt-4o-mini"),
            enable_evolution=True,
            **common_kwargs,
        )
    else:
        memory = MemoryStore(**common_kwargs)

    # Seed memory if requested
    seed_stance = agent_config.get("seed_stance", None)
    if seed_stance:
        topic = global_config["debate"]["topic"]
        seed_review_artifactory(memory, topic, seed_stance, global_config)

    target_initial_stance = agent_config.get("target_initial_stance", None)
    if target_initial_stance is not None and hasattr(memory, "belief_engine"):
        try:
            target_score = float(target_initial_stance)
        except (TypeError, ValueError):
            logger.warning(
                "Ignoring non-numeric target_initial_stance=%r for %s",
                target_initial_stance,
                name,
            )
        else:
            memory.initial_stance = target_score
            if (
                belief_semantics_mode
                == BELIEF_SEMANTICS_ACTIVE_EXPLICIT_PRIOR_V1
            ):
                memory.belief_engine.set_explicit_prior_score(
                    target_score,
                    active_records=memory.arguments.values(),
                    persist=True,
                )
                logger.info(
                    "Initialized %s explicit prior to "
                    "target_initial_stance=%+.3f",
                    name,
                    target_score,
                )
            elif belief_semantics_mode == BELIEF_SEMANTICS_ACTIVE_SEED_V1:
                memory.belief_engine.recompute_from_active_records(
                    memory.arguments.values(),
                    persist=True,
                    increment_history=False,
                )
                logger.info(
                    "Recorded target_initial_stance=%+.3f for %s as metadata; "
                    "active_seed_v1 initializes numerical stance from active "
                    "seed records only",
                    target_score,
                    name,
                )
            elif hasattr(memory.belief_engine, "set_belief_score"):
                memory.belief_engine.set_belief_score(target_score, persist=True)
                logger.info(
                    "Initialized %s latent belief score to target_initial_stance=%+.3f",
                    name,
                    target_score,
                )

    return SubjectAgent(
        backend,
        memory,
        persona,
        name=name,
        temperature=temperature,
        retrieval_k=retrieval_k,
        response_shell=response_shell,
        response_context_messages=response_context_messages,
        response_word_limit=response_word_limit,
        response_retrieval_enabled=response_retrieval_enabled,
        retrieved_evidence_token_budget=retrieved_evidence_token_budget,
        retrieved_evidence_encoding=retrieved_evidence_encoding,
    )


def _create_counter_agent(agent_config, global_config, backend):
    """Create a counter (opposing) agent."""
    from src.agents.counter_agent import CounterAgent
    from src.constants import POLARITY_NEGATIVE

    side = agent_config.get("side", POLARITY_NEGATIVE)
    topic = global_config["debate"]["topic"]
    return CounterAgent(backend, side=side, topic=topic, config=global_config)


def _create_vanilla_agent(agent_config, global_config, backend, name, persona):
    """Create a vanilla agent (no memory, conversation history only)."""
    from src.agents.vanilla_subject_agent import VanillaSubjectAgent

    topic = global_config["debate"]["topic"]
    seed_stance = agent_config.get("seed_stance", "affirmative")
    generation_conf = global_config.get("generation", {})
    temperature = agent_config.get("temperature", generation_conf.get("temperature", 0.7))

    seed_arguments = []
    seeding_limit = global_config.get("debate", {}).get("seeding_limit", 0)
    if seeding_limit > 0:
        from src.simulation.seeding_helper import get_seed_arguments
        seed_arguments = get_seed_arguments(topic, seed_stance, seeding_limit, global_config)

    return VanillaSubjectAgent(
        backend=backend,
        persona=persona,
        name=name,
        topic=topic,
        initial_position=seed_stance,
        seed_arguments=seed_arguments,
        temperature=temperature,
    )


def _create_self_updating_agent(agent_config, global_config, backend, name, persona, log_dir=None):
    """Create a prompt-only self-update baseline, optionally with RAG memory."""
    from src.agents.self_updating_agent import SelfUpdatingSubjectAgent
    from src.memory.memory_store import MemoryStore
    from src.simulation.seeding_helper import get_seed_arguments, seed_review_artifactory

    topic = global_config["debate"]["topic"]
    seed_stance = agent_config.get("seed_stance", "affirmative")
    target_initial_stance = agent_config.get("target_initial_stance", None)
    use_memory = bool(agent_config.get("use_memory", False))
    generation_conf = global_config.get("generation", {})
    memory_conf = global_config.get("memory", {}).copy()
    if "memory" in agent_config:
        memory_conf.update(agent_config["memory"])
    generation_temperature = agent_config.get(
        "temperature",
        generation_conf.get("temperature", 0.7),
    )
    update_temperature = agent_config.get(
        "self_update_temperature",
        generation_conf.get("self_update_temperature", 0.0),
    )
    response_shell = agent_config.get(
        "response_shell",
        generation_conf.get("response_shell", "legacy"),
    )
    response_context_messages = agent_config.get(
        "response_context_messages",
        generation_conf.get("response_context_messages", 4),
    )
    response_word_limit = agent_config.get(
        "response_word_limit",
        generation_conf.get("response_word_limit", 120),
    )
    update_context_messages = agent_config.get(
        "self_update_context_messages",
        generation_conf.get("self_update_context_messages"),
    )
    retrieval_k = agent_config.get(
        "retrieval_k",
        memory_conf.get("retrieval_k", 10),
    )

    seed_arguments = []
    memory = None

    if use_memory:
        common_kwargs = {
            "storage_mode": memory_conf.get("storage_mode", "episodic"),
            "retrieval_strategy": memory_conf.get("retrieval_strategy", "weighted"),
            "worldview_openness": memory_conf.get("worldview_openness", "balanced"),
            "stance_method": memory_conf.get("stance_method", "blind"),
            "update_sensitivity": memory_conf.get("update_sensitivity", 0.0),
            "anchor_sensitivity": memory_conf.get("anchor_sensitivity", 0.0),
            "confirmation_bias": memory_conf.get("confirmation_bias", 0.0),
            "allow_self_arguments": memory_conf.get("allow_self_arguments", True),
            "belief_engine_type": memory_conf.get("belief_engine_type", "bayesian"),
            "belief_semantics_mode": memory_conf.get(
                "belief_semantics_mode",
                "incremental_v1",
            ),
            "backend": backend,
            "agent_id": name,
            "initial_stance": target_initial_stance or 0.0,
            "argument_similarity_threshold": memory_conf.get("argument_similarity_threshold", 0.7),
            "self_similarity_threshold": memory_conf.get("self_similarity_threshold", 0.95),
            "cross_polarity_dedup": memory_conf.get("cross_polarity_dedup", False),
            "deduplication_enabled": memory_conf.get(
                "deduplication_enabled",
                True,
            ),
        }
        if log_dir:
            common_kwargs["memory_dir"] = os.path.join(log_dir, "memory", name)

        memory = MemoryStore(**common_kwargs)
        if seed_stance:
            seed_review_artifactory(memory, topic, seed_stance, global_config)
    else:
        seeding_limit = global_config.get("debate", {}).get("seeding_limit", 0)
        if seeding_limit > 0:
            seed_arguments = get_seed_arguments(topic, seed_stance, seeding_limit, global_config)

    return SelfUpdatingSubjectAgent(
        backend=backend,
        persona=persona,
        name=name,
        topic=topic,
        initial_position=seed_stance,
        initial_stance=target_initial_stance,
        seed_arguments=seed_arguments,
        memory=memory,
        retrieval_k=retrieval_k,
        generation_temperature=generation_temperature,
        update_temperature=update_temperature,
        response_shell=response_shell,
        response_context_messages=response_context_messages,
        response_word_limit=response_word_limit,
        update_context_messages=update_context_messages,
    )


def _create_moltbot_agent(agent_config, global_config, backend, name, persona, log_dir=None):
    """Create a Moltbot agent with belief engine."""
    from src.agents.moltbot_agent import MoltbotAgent

    topic = global_config["debate"]["topic"]
    root_dir = log_dir or global_config.get("logging", {}).get("log_dir", "experiments/logs/moltbot_runs")
    agent_name = agent_config.get("name", "Moltbot")
    belief_engine_type = agent_config.get("belief_engine_type", None)

    kwargs = {
        "topic": topic,
        "root_dir": root_dir,
    }
    if belief_engine_type:
        kwargs["belief_engine_type"] = belief_engine_type

    return MoltbotAgent(backend, persona, agent_name, **kwargs)


def _create_deterministic_agent(agent_config, global_config, name):
    """Create a scripted deterministic agent for controlled experiments."""
    from src.agents.deterministic_agent import DeterministicAgent

    topic = global_config["debate"]["topic"]
    side = agent_config.get("side", "pro")
    seed = global_config["debate"].get("seed", 42)
    rounds = global_config["debate"].get("rounds", 10)

    return DeterministicAgent(
        name=name,
        topic=topic,
        side=side,
        seed=seed,
        limit=rounds * 2,
    )
