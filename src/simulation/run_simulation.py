import argparse
import json
import logging
import os
import shutil
import sys
import glob
import atexit
import yaml
from datetime import datetime
from typing import List, Dict, Any
from src.constants import POLARITY_NEGATIVE

# Global cleanup handler to match Chroma's atexit race condition
def cleanup_artifacts():
    """
    Cleanup ChromaDB temporary directories at exit.
    """
    patterns = ["tempfiles/chroma_db_*"]
    for pattern in patterns:
        for path in glob.glob(pattern):
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    print(f"Cleaned up {path}")
                except Exception as e:
                    print(f"Failed to cleanup {path}: {e}")

atexit.register(cleanup_artifacts)


from src.utils.config import ConfigLoader
from src.utils.logger import ExperimentLogger
from src.backend.factory import create_backend
from src.agents.factory import create_agent
from src.agents.extractor_agent import ArgumentExtractionAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.simulation.simulation_runner import SimulationRunner
from src.utils.result_writer import ResultWriter

logger = logging.getLogger(__name__)

def run_simulation(config_path="config/simulation_default.yaml", log_dir_override=None):
    # 1. Load Config (merges with defaults automatically)
    config = ConfigLoader.load_config(config_path)
    
    # 3. Determine Output Directory (Early)
    # Move this UP so agents can use it for persistent storage paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if config_path:
        config_name = os.path.splitext(os.path.basename(config_path))[0]
    else:
        config_name = "default"
    
    if log_dir_override:
        writer = ResultWriter(log_dir_override)
    else:
        writer = ResultWriter.create_from_config(config_name)
        
    log_dir = writer.get_log_dir()
    
    # Update config so agents effectively see the correct path
    if "logging" not in config:
        config["logging"] = {}
    config["logging"]["log_dir"] = log_dir
    
    # Save resolved config for reproducibility
    writer.save_yaml("resolved_config.yaml", config)

    # 4. Setup Backend
    backend = create_backend(config)
    
    # Setup Extractor (Shared)
    topic = config["debate"]["topic"]
    extractor = ArgumentExtractionAgent(backend, topic_id=topic)
    
    # 5. Create Agents
    agents = []
    sim_config = config.get("simulation", {})
    agent_configs = sim_config.get("agents", [])
    
    # Fallback to default 1v1 if no agents defined
    if not agent_configs:
        logger.warning("No agents defined in 'simulation' config section. Falling back to Default Subject vs Counter.")

        # Build Subject agent config with seeding parameters from debate config
        subject_config = {
            "name": "Subject",
            "type": config["debate"].get("agent_type", "subject"),
            "persona": config["debate"]["subject_persona"]
        }

        # Copy seeding parameters from debate config if present
        if "seed_stance" in config["debate"]:
            subject_config["seed_stance"] = config["debate"]["seed_stance"]
        if "target_initial_stance" in config["debate"]:
            subject_config["target_initial_stance"] = config["debate"]["target_initial_stance"]

        agent_configs = [
            subject_config,
            {
                "name": "Counter",
                "type": "counter",
                "side": POLARITY_NEGATIVE
            }
        ]
        
    for agent_conf in agent_configs:
        try:
            agent = create_agent(agent_conf, config, backend, log_dir=log_dir)
            agents.append(agent)
            logger.info(f"Created Agent: {agent.name} ({agent_conf['type']})")
        except Exception as e:
            logger.error(f"Failed to create agent {agent_conf.get('name')}: {e}")

    # Seed memory if configured
    seeding_t = config.get("debate", {}).get("seeding_time")
    if seeding_t == "pre_simulation":
        logging.info("Starting memory seeding phase...")
        
        strategy = config.get("debate", {}).get("seeding_strategy", "quality_max")
        
        if strategy == "score_matched":
            from src.simulation.seeding_helper import seed_agents_collaboratively
            seed_agents_collaboratively(agents, agent_configs, config.get("debate", {}).get("topic"), config)
        else:
            from src.simulation.seeding_helper import seed_review_artifactory
            for agent, agent_conf in zip(agents, agent_configs):
                # Skip vanilla agents (no memory)
                if getattr(agent, "memory", None) is None:
                    continue
                start_stance = agent_conf.get("seed_stance")
                if start_stance:
                    seed_review_artifactory(agent.memory, topic, start_stance, config)

    # Setup Evaluator (Shared)
    evaluator = None
    if config.get("evaluation", {}).get("enabled", True): # Default to True for backward compat
        evaluator = EvaluationAgent(backend)

    # 4. Run Simulation
    # 6. Run Simulation
    try:
        rounds = sim_config.get("rounds", config["debate"]["rounds"])
        # Pass extractor to enable argument exchange AND log_dir (already calculated)
        logging_config = config.get("logging", {})
        artifact_mode = logging_config.get("artifact_mode", "full")
        live_plot = bool(logging_config.get("live_plot", False) or sim_config.get("live_plot", False))
        runner = SimulationRunner(
            agents,
            rounds=rounds,
            extractor=extractor,
            evaluator=evaluator,
            output_dir=log_dir,
            live_plot=live_plot,
            live_json=bool(logging_config.get("live_json", artifact_mode != "compact")),
            artifact_mode=artifact_mode,
            evaluator_strict=bool(
                config.get("evaluation", {}).get("strict_parsing", False)
            ),
            speaker_order=sim_config.get("speaker_order", "pro-first"),
        )
        results = runner.run()
        
        # 5. Save Results
        writer.save_json("results.json", results)
        writer.save_memory_dump(agents)
        if artifact_mode != "compact":
            writer.save_settings_summary(config_name, config_path, config)
            
        logger.info(f"Simulation complete. Results saved to {log_dir}")
        
        # 6. Generate Plot
        try:
            if artifact_mode != "compact":
                from scripts.analysis.plot_simulation import plot_simulation_results
                plot_simulation_results(log_dir)
        except Exception as e:
            logger.error(f"Failed to generate plot: {e}")

        return results

    finally:
        # 6. Cleanup Agents
        logger.info("Cleaning up agents...")
        for agent in agents:
            if hasattr(agent, "close"):
                try:
                    agent.close()
                except Exception as e:
                    logger.warning(f"Error closing agent {agent.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a multi-agent simulation.")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--sim", type=str, help="Name of simulation from registry (config/simulations/)")
    parser.add_argument("--list", action="store_true", help="List available simulations in registry")
    parser.add_argument("--log-dir", type=str, help="Override output log directory")
    args = parser.parse_args()
    
    registry_dir = os.path.join("config", "simulations")
    
    if args.list:
        print(f"\nAvailable Simulations in '{registry_dir}':")
        if os.path.exists(registry_dir):
            for f in sorted(os.listdir(registry_dir)):
                if f.endswith(".yaml") or f.endswith(".yml"):
                    print(f"  - {os.path.splitext(f)[0]}")
        else:
            print("  (Registry directory not found)")
        print("")
        sys.exit(0)

    config_path = args.config
    
    if args.sim:
        # Resolve from registry
        candidate = os.path.join(registry_dir, f"{args.sim}.yaml")
        if os.path.exists(candidate):
            config_path = candidate
        else:
            # Try without extension or .yml
            candidate_yml = os.path.join(registry_dir, f"{args.sim}.yml")
            if os.path.exists(candidate_yml):
                config_path = candidate_yml
            else:
                print(f"Error: Simulation '{args.sim}' not found in {registry_dir}")
                sys.exit(1)
    
    if not config_path:
        # Default fallback
        config_path = "config/defaults.yaml"
        print(f"No config specified, using default: {config_path}")

    run_simulation(config_path, log_dir_override=args.log_dir)
