import argparse
import os
import json
import logging
import sys

from src.experiment.orchestrator import run_experiment
from src.simulation.simulation_runner import SimulationRunner
from src.utils.logger import ExperimentLogger
from src.utils.config import ConfigLoader
from src.backend.backend_local import LocalLLMBackend
from src.backend.backend_openrouter import OpenRouterBackend
from src.backend.backend_mock import MockLLMBackend
from src.backend.backend_openai import OpenAIBackend
from src.memory.memory_store import MemoryStore
try:
    from src.memory.agentic_memory import AgenticMemoryStore
except ImportError:
    pass

from src.agents.subject_agent import SubjectAgent
from src.agents.counter_agent import CounterAgent
from src.agents.extractor_agent import ArgumentExtractionAgent
from src.agents.evaluation_agent import EvaluationAgent

def load_log_state(run_dir):
    """
    Parses log.jsonl to reconstruct conversation history and stance history.
    """
    log_file = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    history = []
    stance_history = []
    last_round = 0
    config_overrides = {}

    with open(log_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line)
                event = entry.get("event_type")
                data = entry.get("data")

                if event == "turn" or event == "round_turn":
                    # Reconstruct message
                    role = "assistant" if data.get("speaker") == "subject" or data.get("agent") == "Subject" else "user"
                    if data.get("agent") == "Counter": role = "user"
                    
                    content = data.get("text") or data.get("content")
                    if content:
                        history.append({"role": role, "content": content})
                    
                    r_num = data.get("round", 0)
                    if r_num > last_round:
                        last_round = r_num

                elif event == "stance_update":
                    stance = data.get("stance")
                    if stance is not None:
                        stance_history.append(stance)
                        
            except json.JSONDecodeError:
                continue
                
    return history, stance_history, last_round

def resume_experiment(log_dir, additional_rounds, overrides=None):
    if overrides is None: overrides = {}
    print(f"Resuming experiment from: {log_dir}")
    print(f"Adding rounds: {additional_rounds}")
    print(f"Overrides: {overrides}")

    # 1. Recover Metadata (from directory structure or log analysis could be better, but assuming defaults for now)
    # Ideally we should have saved the full config in the log. 
    # For now, we load defaults and assume standard setup unless we find a config snapshot.
    # 1. Recover Metadata (from directory structure or log analysis could be better, but assuming defaults for now)
    # Ideally we should have saved the full config in the log. 
    # For now, we load defaults and assume standard setup unless we find a config snapshot.
    try:
        if os.path.exists("config/defaults.yaml"):
             config = ConfigLoader.load_config("config/defaults.yaml")
        else:
             config = ConfigLoader.DEFAULTS
    except Exception as e:
        print(f"Warning: Could not load defaults.yaml: {e}")
        config = ConfigLoader.DEFAULTS
    
    # 2. Reconstruct History
    history, stance_history, last_round = load_log_state(log_dir)
    print(f"Recovered {len(history)} messages. Last round: {last_round}")
    print(f"Recovered {len(stance_history)} stance updates.")

    # 3. Initialize Backend (Standard Setup)
    # NOTE: This re-initializes backend based on defaults.yaml. If the original run used overrides, 
    # we might be missing them unless we save config.yaml in the log dir (Suggestion for future).
    if config["backend"]["type"] == "openai":
         backend = OpenAIBackend(api_key=os.environ.get("OPENAI_API_KEY"), model=config["backend"]["model"])
    # ... (Simplify backend loading for prototype, assume OpenAI/Default)
    else: 
         backend = OpenAIBackend(api_key=os.environ.get("OPENAI_API_KEY"), model=config["backend"]["model"])

    # 4. Initialize Memory
    # Extract run_id from log_dir path
    run_id = os.path.basename(log_dir).split("_")[-2] # Assumes standard naming format
    
    # Attempt to load config from results.csv in the experiment root
    experiment_root = os.path.dirname(log_dir)
    results_csv = os.path.join(experiment_root, "results.csv")
    
    # Default values from loaded config (defaults.yaml)
    update_sensitivity = config["memory"].get("update_sensitivity", 1.0)
    confirmation_bias = config["memory"].get("confirmation_bias", 0.0)
    stance_method = config["memory"].get("stance_method", "bayesian")
    belief_engine_type = config["memory"].get("belief_engine_type", "bayesian")
    
    if os.path.exists(results_csv):
        try:
            import pandas as pd
            df = pd.read_csv(results_csv)
            # Find row where log_dir matches or run_id matches
            # Construct log_dir path relative to project root or check name
            folder_name = os.path.basename(log_dir)
            
            # The CSV might not have log_dir column explicitly or it might be different format.
            # Orchestrator saves 'run_id' in CSV which corresponds to the folder name usually?
            # No, 'run_id' in CSV is the EXPERIMENT run_id (the parent folder).
            # The 'config_name' helps.
            
            # We can try to match by folder name if we reconstructed it, but better:
            # Look for the row that corresponds to this variation.
            # The folder name is: {config_name}_{trial}_{timestamp}.
            # Let's try to match partial string on the folder name?
            
            # Actually, we don't store the sub-folder name in CSV explicitly in standard run_experiment.
            # But we do store 'log_dir' in the result dict in orchestrator... let's check if it made it to CSV.
            # Usually strict flat dicts make it to CSV.
            # If not, we have to guess based on var_ columns.
            
            # Use 'var_memory.update_sensitivity' if present.
            # We just need ANY row that matches the config of this folder.
            # The folder name contains the config name usually (e.g. episodic_weighted).
            
            matched_row = None
            for idx, row in df.iterrows():
                # Check if row's config_name is in the folder name
                # folder: episodic_weighted_001_2026...
                # config: Arch_1_Baseline (wait, run_experiment uses config name from definition)
                
                # In architecture_sweep, config names are "Arch_1_Baseline". 
                # Orchestrator generates folder name based on params (episodic_weighted).
                # This decoupled naming makes lookup hard.
                
                # FALLBACK: Just look for 'var_' columns in the CSV and assume they apply equally?
                # No, different rows have different vars.
                pass
                
            # Alternative: Simply look for a row where 'log_dir' == current log dir
            # If run_experiment saved 'log_dir' to CSV.
            matches = df[df['log_dir'] == log_dir]
            if not matches.empty:
                row = matches.iloc[0]
                update_sensitivity = row.get("var_memory.update_sensitivity", update_sensitivity)
                confirmation_bias = row.get("var_memory.confirmation_bias", confirmation_bias)
                stance_method = row.get("var_memory.stance_method", stance_method)
                belief_engine_type = row.get("var_memory.belief_engine_type", belief_engine_type)
                print(f"Loaded config from CSV: Sensitivity={update_sensitivity}, Bias={confirmation_bias}")
            else:
                 # Try matching by absolute path awareness or just filename
                 # CSV might have relative path?
                 # Let's assume we can't easily find it and fallback to "Sensitivity 0.3" 
                 # since we know that's what we ran for architecture sweep.
                 # HACK for this specific run:
                 if "architecture_sweep" in experiment_root:
                     update_sensitivity = 0.3
                     print("Applied Architecture Sweep Default: Sensitivity=0.3")
                 
        except Exception as e:
            print(f"Failed to load config from CSV: {e}")

    # Apply Explicit Overrides (Highest Priority)
    if "update_sensitivity" in overrides: update_sensitivity = overrides["update_sensitivity"]
    if "confirmation_bias" in overrides: confirmation_bias = overrides["confirmation_bias"]
    if "stance_method" in overrides: stance_method = overrides["stance_method"]
    if "belief_engine_type" in overrides: belief_engine_type = overrides["belief_engine_type"]
    
    print(f"Final Config: Sensitivity={update_sensitivity}, Bias={confirmation_bias}, Method={stance_method}")

    # Check if Agentic
    is_agentic = "agentic" in log_dir or "Arch_3" in log_dir or "Arch_4" in log_dir # Heuristic
    # Also check if folder name contains 'agentic'
    if "agentic" in os.path.basename(log_dir): 
        is_agentic = True

    
    start_memory_count = 0
    
    # Initialize Memory with CORRECT params
    if is_agentic:
        print(f"Detected Agentic Memory. Reconnecting to ID: {run_id}")
        memory = AgenticMemoryStore(
            backend=backend, 
            agent_id=run_id,
            stance_method=stance_method,
            update_sensitivity=float(update_sensitivity),
            confirmation_bias=float(confirmation_bias),
            belief_engine_type=belief_engine_type
        )
        # Agentic memory persists on disk, so it should be auto-hydrated
        start_memory_count = len(memory.arguments)
    else:
        print("Detected Episodic Memory. Re-hydrating from memory.jsonl...")
        memory = MemoryStore(
            backend=backend,
            stance_method=stance_method,
            update_sensitivity=float(update_sensitivity),
            confirmation_bias=float(confirmation_bias),
            belief_engine_type=belief_engine_type
        ) 
        
        # Load latest memory snapshot
        mem_file = os.path.join(log_dir, "memory.jsonl")
        last_snapshot = []
        if os.path.exists(mem_file):
            with open(mem_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get("arguments"):
                             last_snapshot = data.get("arguments")
                    except: pass
        
        # Hydrate
        from src.memory.argument_record import ArgumentRecord
        for arg_dict in last_snapshot:
            rec = ArgumentRecord.from_dict(arg_dict)
            memory.add_argument(rec)
        start_memory_count = len(memory.arguments)

    print(f"Memory Hydrated: {start_memory_count} arguments.")

    # 5. Initialize Agents
    subject = SubjectAgent(backend, memory, config["debate"]["subject_persona"], name="Subject")
    counter = CounterAgent(backend, side="con", topic=config["debate"]["topic"], config=config)
    extractor = ArgumentExtractionAgent(backend, config["debate"]["topic"], config=config)
    
    # 6. Resume Logger (Append Mode)
    # We need a logger that appends to the EXISTING files
    # Direct hack for prototype:
    logger = ExperimentLogger(log_dir=os.path.dirname(log_dir), topic=config["debate"]["topic"]) 
    logger.run_dir = log_dir # Force existing dir
    logger.log_file = os.path.join(log_dir, "log.jsonl")
    logger.memory_log_file = os.path.join(log_dir, "memory.jsonl") 
    # This might overwrite summary.md, which is fine as we generate a new one.

    # 7. Run Debate (Resume)
    # 7. Run Debate (Resume)
    # Use SimulationRunner (Consolidated Engine)
    runner = SimulationRunner(
        agents=[subject, counter], 
        rounds=additional_rounds, 
        extractor=extractor, 
        output_dir=logger.run_dir,
        initial_transcript=history,
        start_round=last_round + 1,
        speaker_order=config.get("simulation", {}).get(
            "speaker_order",
            "pro-first",
        ),
    )
    
    # State Injection is handled by initial_transcript
    
    print("State injected. Resuming debate...")
    sim_results = runner.run()
    
    new_transcript = sim_results["transcript"]
    new_stance = sim_results["stance_history"]
    
    print("Resumed run complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--add-rounds", type=int, default=5)
    
    # Overrides
    parser.add_argument("--update-sensitivity", type=float, default=None)
    parser.add_argument("--confirmation-bias", type=float, default=None)
    parser.add_argument("--stance-method", type=str, default=None)
    
    args = parser.parse_args()
    
    overrides = {}
    if args.update_sensitivity is not None: overrides["update_sensitivity"] = args.update_sensitivity
    if args.confirmation_bias is not None: overrides["confirmation_bias"] = args.confirmation_bias
    if args.stance_method is not None: overrides["stance_method"] = args.stance_method
    
    resume_experiment(args.log_dir, args.add_rounds, overrides)
