import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def run_evaluation(
    config: Dict[str, Any],
    exp_logger,
    evaluator,
    memory,
    subject,
    sim_results: Dict[str, Any],
    initial_memory_snapshot: List[Dict[str, Any]],
    seeded_stance: float,
    topic_name: str
) -> Dict[str, Any]:
    """
    Handles pre- and post-debate qualitative evaluation, quantitative extraction,
    and markdown summary generation for the orchestrator.
    """
    transcript = sim_results.get("transcript", [])
    stance_history = sim_results.get("stance_history", [])
    
    # 1. Extract Opening Statement
    opening_statement = "No opening statement found."
    if transcript and len(transcript) > 0:
        if transcript[0]['role'] == 'assistant':
            opening_statement = transcript[0]['content']
    
    # 2. Capture Initial Stance
    initial_stance = subject.get_current_stance()
    if stance_history:
        initial_stance = stance_history[0].get(subject.name, initial_stance)
        
    exp_logger.log_event("initial_stance_post_opening", {"stance": initial_stance})
    
    # 3. Initial Qualitative Evaluation
    initial_memory_context = str(initial_memory_snapshot)
    initial_eval_result = evaluator.evaluate_stance(
        f"Persona: {config['debate']['subject_persona']}\nOpening Statement: {opening_statement}", 
        initial_memory_context,
        topic=topic_name,
        agent_name="Subject"
    )
    print(f"Evaluator Estimated Initial Stance (LLM-Based): {initial_eval_result.get('stance_score')}")
    exp_logger.log_event("pre_eval_qualitative", {
        "initial_qualitative_stance": initial_eval_result.get("stance_score")
    })

    # 4. Post-Debate Setup
    logger.info("Running post-debate evaluation...")
    final_stance = subject.get_current_stance()
    
    role_map = {"user": "Counter Agent", "assistant": "Subject Agent"}
    transcript_text = "\n".join([f"{role_map.get(msg['role'], msg['role'])}: {msg['content']}" for msg in transcript])
    memory_snapshot = str(memory.get_all_arguments() if memory else [])
    
    # 5. Extract Final Statement
    final_voter_msg = "No final statement found."
    for msg in reversed(transcript):
        if msg['role'] == "assistant":
            final_voter_msg = msg['content']
            break
            
    # 6. Post-Debate Qualitative Evaluation
    stance_instruction = None
    if memory and hasattr(memory, 'belief_engine') and memory.belief_engine:
        stance_instruction = memory.belief_engine.get_belief_description()
            
    eval_result = evaluator.evaluate_stance(
        f"Final Statement: {final_voter_msg}", 
        memory_snapshot, 
        topic=topic_name, 
        agent_name="Subject",
        expected_stance_instruction=stance_instruction
    )
    consistency_result = evaluator.evaluate_consistency(config["debate"]["subject_persona"], transcript_text)
    summary = evaluator.summarize_debate(transcript_text, initial_stance, final_stance)

    # 7. Final Metrics
    final_prob = memory.get_bayesian_probability() if memory and hasattr(memory, "get_bayesian_probability") else None
    prob_str = f" (Prob: {final_prob:.4f})" if final_prob is not None else ""

    exp_logger.log_event("post_eval", {
        "final_calculated_stance": final_stance,
        "final_bayesian_probability": final_prob,
        "evaluator_estimated_stance": eval_result.get("stance_score"),
        "consistency": consistency_result,
        "summary": summary
    })

    logger.info(f"Final Quantitative Stance (Calculated): {final_stance}{prob_str}")
    logger.info(f"Final Qualitative Stance (LLM-Based): {eval_result.get('stance_score')}")
    
    # 8. Calculate Memory Changes
    final_memory = memory.get_all_arguments() if memory else []
    initial_claims = {arg.get('claim', '') for arg in initial_memory_snapshot}
    
    new_arguments = []
    for arg in final_memory:
        if arg.get('claim', '') not in initial_claims:
            new_arguments.append(arg)

    memory_summary = f"\n\n## Memory Evolution\n"
    memory_summary += f"- **Initial Arguments**: {len(initial_memory_snapshot)}\n"
    memory_summary += f"- **Final Arguments**: {len(final_memory)}\n"
    memory_summary += f"- **New Arguments Acquired**: {len(new_arguments)}\n\n"
    
    if new_arguments:
        memory_summary += "### Newly Acquired Arguments:\n"
        for arg in new_arguments:
            polarity = str(arg.get('polarity', 'unknown')).upper()
            claim = arg.get('claim', 'No claim text')
            strength = arg.get('support_strength', 0.0)
            memory_summary += f"- [{polarity}] {claim} (Strength: {strength:.2f})\n"

    # 9. Format Final Markdown Summary
    mode_descriptions = {
        "episodic": "Stores every argument as a distinct record.",
        "worldview": "Stores arguments with an additional worldview-consistency check.",
        "agentic": "Uses the optional Chroma-backed experimental memory store.",
    }
    storage_mode = config.get("memory", {}).get("storage_mode", "episodic")
    mode_desc = mode_descriptions.get(storage_mode, "Unknown mode")

    stance_method_label = config.get("memory", {}).get("stance_method", "linear")
    belief_engine_type = config.get("memory", {}).get("belief_engine_type", None)
    if belief_engine_type in ("agentic", "llm"):
        stance_method_label = "agentic"

    full_summary = f"""# Experiment Summary: {topic_name}

## Experiment Configuration
- **Run ID**: {exp_logger.run_id}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Rounds**: {config.get("debate", {}).get("rounds", 10)}
- **Seed Stance**: {config.get("debate", {}).get("seed_stance", "none")}
- **Subject Persona**: {config.get("debate", {}).get("subject_persona", "N/A")}
- **Memory Decay Factor**: {config.get("memory", {}).get("decay_factor", 1.0)}
- **Retrieval Strategy**: {config.get("memory", {}).get("retrieval_strategy", "mixed")}
- **Storage Mode**: {storage_mode}
  - *{mode_desc}*
- **Worldview Openness**: {config.get("memory", {}).get("worldview_openness", "balanced")} (Only for 'worldview' mode)
- **Stance Method**: {stance_method_label}
- **Update Sensitivity**: {config.get("memory", {}).get("update_sensitivity", "N/A")}
- **Anchor Sensitivity**: {config.get("memory", {}).get("anchor_sensitivity", "N/A")}
- **Confirmation Bias**: {config.get("memory", {}).get("confirmation_bias", "0.0")}

## Stance Evolution
- **Seeded Stance**: {seeded_stance:.2f}
- **Quantitative Stance (Calculated)**: {initial_stance:.2f} -> {final_stance:.2f} (Shift: {final_stance - initial_stance:.2f})
- **Final Bayesian Probability**: {final_prob if final_prob is not None else "N/A"} (0.0 = Against, 1.0 = In Favor)
- **Qualitative Stance (LLM-Estimated)**: {initial_eval_result.get('stance_score')} -> {eval_result.get('stance_score')}
  - *Estimated by an external LLM evaluator reading the debate transcript.*

## Narrative Summary
{summary}
"""
    full_summary += memory_summary

    exp_logger.log_summary(full_summary)
    
    print("\n" + "="*50)
    print("DEBATE SUMMARY")
    print("="*50)
    print(summary)
    print("="*50 + "\n")
    
    return {
        "initial_stance": initial_stance,
        "final_stance": final_stance,
        "eval_result": eval_result,
        "final_memory": final_memory,
        "new_arguments": new_arguments
    }
