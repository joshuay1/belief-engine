import os
import sys
import copy
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.config import ConfigLoader
from src.utils.logger import ExperimentLogger
from src.backend.backend_local import LocalLLMBackend
from src.backend.backend_openrouter import OpenRouterBackend
from src.backend.backend_mock import MockLLMBackend
from src.backend.backend_openai import OpenAIBackend
from src.memory.memory_store import MemoryStore
try:
    from src.memory.agentic_memory import AgenticMemoryStore
except ImportError as e:
    # Dummy class to prevent ImportErrors if chromadb is missing
    class AgenticMemoryStore:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"AgenticMemoryStore requires 'chromadb' to be installed. Error: {e}")
from src.memory.argument_record import ArgumentRecord
import json
from src.agents.subject_agent import SubjectAgent
from src.agents.counter_agent import CounterAgent
from src.agents.extractor_agent import ArgumentExtractionAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.simulation.simulation_runner import SimulationRunner
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_DISPLAY, POLARITY_MAPPING

def print_seeding_similarity_stats(memory, credence_manager):
    """
    Calculate and print pairwise cosine similarity statistics for seeded arguments.
    
    Args:
        memory: MemoryStore instance
        credence_manager: CredenceManager instance for calculating similarities
    """
    all_memory_args = list(memory.arguments.values())
    
    if len(all_memory_args) > 1:
        # Get embeddings for all arguments
        embeddings = []
        for arg in all_memory_args:
            embedding = credence_manager._get_embedding(arg.claim)
            embeddings.append(embedding)
        
        # Calculate pairwise similarities (upper triangle only to avoid duplicates)
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = credence_manager.cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            min_similarity = min(similarities)
            max_similarity = max(similarities)
            
            print(f"\n{'='*50}")
            print("SEEDING SIMILARITY STATISTICS")
            print(f"{'='*50}")
            print(f"Total arguments seeded: {len(all_memory_args)}")
            print(f"Pairwise comparisons: {len(similarities)}")
            print(f"Average pairwise cosine similarity: {avg_similarity:.4f}")
            print(f"Minimal pairwise cosine similarity: {min_similarity:.4f}")
            print(f"Maximal pairwise cosine similarity: {max_similarity:.4f}")
            print(f"{'='*50}\n")
    else:
        print(f"\n{'='*50}")
        print("SEEDING SIMILARITY STATISTICS")
        print(f"{'='*50}")
        print(f"Total arguments seeded: {len(all_memory_args)}")
        print("Not enough arguments for pairwise comparison (need at least 2)")
        print(f"{'='*50}\n")

def run_experiment(config_overrides=None, return_results=False, batch_dir=None, run_name_prefix=None):
    # 1. Load Config
    # 1. Load Config
    try:
        config = ConfigLoader.load_config("config/defaults.yaml")
    except Exception as e:
        print(f"Warning: Could not load defaults.yaml: {e}")
        config = ConfigLoader.DEFAULTS
    
    # Apply overrides
    if config_overrides:
        # Deep merge simple overrides
        for key, value in config_overrides.items():
            if "." in key:
                # Split on FIRST dot only, allowing for sections like 'memory' to have subkeys
                parts = key.split(".", 1)
                if len(parts) == 2:
                   section, subkey = parts
                   if section in config:
                       config[section][subkey] = value
                else:
                    # Should not happen with split(., 1) but good safety
                    pass
            else:
                config[key] = value

    topic_name = config["debate"]["topic"]
    storage_mode = config.get("memory", {}).get("storage_mode", "episodic")
    worldview_openness = config.get("memory", {}).get("worldview_openness", "balanced")
    
    # Only include openness if mode is worldview
    openness_val = worldview_openness if storage_mode == "worldview" else None
    
    retrieval_strategy = config.get("memory", {}).get("retrieval_strategy", "weighted")
    
    logger = ExperimentLogger(config["logging"]["log_dir"], topic=topic_name, mode=storage_mode, openness=openness_val, retrieval_strategy=retrieval_strategy, batch_dir=batch_dir, run_name_prefix=run_name_prefix)
    logger.info("Configuration loaded.")

    # 2. Setup Backend
    if config["backend"]["type"] == "local":
        backend = LocalLLMBackend(
            base_url=config["backend"]["url"],
            api_key=config["backend"]["api_key"],
            model=config["backend"]["model"]
        )
    elif config["backend"]["type"] == "mock":
        backend = MockLLMBackend()
    elif config["backend"]["type"] == "openai":
        api_key = config["backend"].get("api_key")
        env_key = os.environ.get("OPENAI_API_KEY")
        
        # If config key is dummy/missing, check env
        if not api_key or api_key in ["sk-dummy", "sk-example"]:
            if env_key and not env_key.startswith("sk-dummy"):
                # Real key found in env, use it
                api_key = env_key
                # Only use config URL if it's NOT localhost (to avoid overriding real API with default local config)
                url = config["backend"].get("url", "")
                if "localhost" not in url and "127.0.0.1" not in url:
                    os.environ.setdefault('OPENAI_BASE_URL', url)
            else:
                # No real key, assume local LLM and use config URL
                os.environ.setdefault('OPENAI_BASE_URL', config["backend"]["url"])
                os.environ.setdefault('OPENAI_API_KEY', api_key or "sk-dummy")
                api_key = os.environ.get("OPENAI_API_KEY", api_key)
            
        backend = OpenAIBackend(
            api_key=api_key,
            model=config["backend"]["model"]
        )
    else:
        backend = OpenRouterBackend(
            api_key=config["backend"]["api_key"],
            model=config["backend"]["model"]
        )

    # Optional: Use a separate model for the evaluator
    evaluator_backend = backend
    eval_config = config.get("evaluation", {})
    eval_model = eval_config.get("model")
    if eval_model and eval_model != config["backend"]["model"]:
        if config["backend"]["type"] == "local":
            evaluator_backend = LocalLLMBackend(
                base_url=config["backend"]["url"],
                api_key=config["backend"]["api_key"],
                model=eval_model
            )
        elif config["backend"]["type"] == "mock":
            evaluator_backend = MockLLMBackend()
        elif config["backend"]["type"] == "openai":
            api_key = config["backend"].get("api_key")
            env_key = os.environ.get("OPENAI_API_KEY")

            if not api_key or api_key in ["sk-dummy", "sk-example"]:
                if env_key and not env_key.startswith("sk-dummy"):
                    api_key = env_key
                    url = config["backend"].get("url", "")
                    if "localhost" not in url and "127.0.0.1" not in url:
                        os.environ.setdefault('OPENAI_BASE_URL', url)
                else:
                    os.environ.setdefault('OPENAI_BASE_URL', config["backend"]["url"])
                    os.environ.setdefault('OPENAI_API_KEY', api_key or "sk-dummy")
                    api_key = os.environ.get("OPENAI_API_KEY", api_key)

            evaluator_backend = OpenAIBackend(
                api_key=api_key,
                model=eval_model
            )
        else:
            evaluator_backend = OpenRouterBackend(
                api_key=config["backend"]["api_key"],
                model=eval_model
            )

    # Initialize Memory Store
    memory_config = config.get("memory", {})
    storage_mode = memory_config.get("storage_mode", "episodic")
    retrieval_strategy = memory_config.get("retrieval_strategy", "weighted")
    worldview_openness = memory_config.get("worldview_openness", 0.5)
    store = AgenticMemoryStore if storage_mode == "agentic" else MemoryStore

    stance_method = memory_config.get("stance_method", "linear")
    belief_engine_type = memory_config.get("belief_engine_type", "bayesian")
    update_sensitivity = memory_config.get("update_sensitivity", 2.0)
    anchor_sensitivity = memory_config.get("anchor_sensitivity", None)
    confirmation_bias = memory_config.get("confirmation_bias", 0.0)
    argument_similarity_threshold = memory_config.get("argument_similarity_threshold", 0.7)
    seeding_random_seed = config["debate"].get("seeding_random_seed", None)
    
    # Feature: Auto-calculate anchor_sensitivity from target_initial_stance
    target_initial_stance = config["debate"].get("target_initial_stance", None)
    if target_initial_stance is not None and target_initial_stance != "none":
        # Only works if we know the seeding limit and assume strength~=1.0 for seeds
        seeding_limit = config["debate"].get("seeding_limit", 10)
        
        try:
            target_stance = float(target_initial_stance)
            if target_stance >= 1.0: target_stance = 0.999 # Avoid infinity
            if target_stance <= -1.0: target_stance = -0.999
            
            # 1. Convert Target Stance to Probability
            # Stance = (P * 2) - 1  => P = (Stance + 1) / 2
            target_prob = (abs(target_stance) + 1.0) / 2.0
            
            # 2. Convert to Log Odds
            import math
            target_log_odds = math.log(target_prob / (1.0 - target_prob))
            
            # 3. Log contribution per argument
            # L_total = N * L_arg
            # L_arg = L_total / N
            log_arg = target_log_odds / seeding_limit
            
            # 4. Solve for Sensitivity
            # L_arg = ln(1 + strength * K)
            # e^L_arg = 1 + strength * K
            # K = (e^L_arg - 1) / strength
            # Assume strength = 1.0 for optimal seeds (quality_max)
            assumed_strength = 1.0
            
            calculated_sensitivity = (math.exp(log_arg) - 1.0) / assumed_strength
            
            logger.info(f"Target Stance {target_stance} requires Anchor Sensitivity: {calculated_sensitivity:.4f} (based on {seeding_limit} seeds)")
            anchor_sensitivity = calculated_sensitivity
            
        except Exception as e:
            logger.error(f"Failed to calculate anchor sensitivity from target stance: {e}")

    unique_agent_id = os.path.basename(logger.run_dir)
    memory_dir = os.path.join(logger.run_dir, "memory")

    allow_self_arguments = memory_config.get("allow_self_arguments", None)

    # Save the effective runtime config into the trial log directory so the run
    # can be reproduced directly from its artifacts.
    resolved_config = copy.deepcopy(config)
    resolved_config.setdefault("memory", {})
    resolved_config.setdefault("debate", {})
    resolved_config["memory"]["storage_mode"] = storage_mode
    resolved_config["memory"]["retrieval_strategy"] = retrieval_strategy
    resolved_config["memory"]["worldview_openness"] = worldview_openness
    resolved_config["memory"]["stance_method"] = stance_method
    resolved_config["memory"]["belief_engine_type"] = belief_engine_type
    resolved_config["memory"]["update_sensitivity"] = update_sensitivity
    resolved_config["memory"]["anchor_sensitivity"] = anchor_sensitivity
    resolved_config["memory"]["confirmation_bias"] = confirmation_bias
    resolved_config["memory"]["argument_similarity_threshold"] = argument_similarity_threshold
    resolved_config["memory"]["allow_self_arguments"] = allow_self_arguments
    resolved_config["debate"]["target_initial_stance"] = target_initial_stance
    resolved_config["debate"]["seeding_random_seed"] = seeding_random_seed
    resolved_config["logging"]["log_dir"] = logger.run_dir

    from src.utils.result_writer import ResultWriter
    result_writer = ResultWriter(logger.run_dir)
    result_writer.save_yaml("resolved_config.yaml", resolved_config)
    logger.log_event("config_snapshot", resolved_config)

    memory = store(
        storage_mode=storage_mode, 
        retrieval_strategy=retrieval_strategy,
        worldview_openness=worldview_openness,
        stance_method=stance_method,
        update_sensitivity=update_sensitivity,
        anchor_sensitivity=anchor_sensitivity,
        confirmation_bias=confirmation_bias,
        argument_similarity_threshold=argument_similarity_threshold,
        self_similarity_threshold=memory_config.get("self_similarity_threshold", 0.95),
        cross_polarity_dedup=memory_config.get("cross_polarity_dedup", False),
        backend=backend,
        agent_id=unique_agent_id,
        memory_dir=memory_dir,
        allow_self_arguments=allow_self_arguments if allow_self_arguments is not None else True,
        belief_engine_type=belief_engine_type
    )


    agent_type = config["debate"].get("agent_type", "subject")
    
    if "moltbot" in agent_type:
        from src.agents.moltbot_agent import MoltbotAgent, MotlbotStanceAgent
        logger.info("Instantiating MoltbotAgent...")
        if "stance" in agent_type: # moltbot_stance
            subject = MotlbotStanceAgent(
                backend,
                config["debate"]["subject_persona"],
                name="Subject",
                topic=topic_name,
                root_dir=logger.run_dir,
                belief_engine_type=belief_engine_type
            )
        else:
            subject = MoltbotAgent(
                backend,
                config["debate"]["subject_persona"],
                name="Subject",
                topic=topic_name,
                root_dir=logger.run_dir,
                belief_engine_type=belief_engine_type
            )
        # Swap memory reference for compatibility with logging/seeding
        memory = subject.memory
        if hasattr(memory, "allow_self_arguments"):
            if allow_self_arguments is not None:
                memory.allow_self_arguments = allow_self_arguments
        
    elif "deterministic" in agent_type:
        from src.agents.deterministic_agent import DeterministicAgent
        logger.info("Instantiating DeterministicAgent...")
        seed = config["debate"].get("seed", 42)
        # For seed, use trial number ideally? Or config seed.
        # Orchestrator doesn't know trial! 'config' comes from run_experiment override.
        # run_experiment.py passes 'var_seed' if varied.
        # If seed is not varied, it's fixed.
        # To vary per trial, we need 'trial' passed to orchestrator?
        # SimulationRunner handles randomness internally usually? No.
        # Let's use config.get("seed", 42). If user wants variety, they vary seed in config.
        
        subject = DeterministicAgent(
            name="Subject", 
            topic=topic_name, 
            side="pro", # Assume subject is Pro for now, or read from config
            seed=seed,
            limit=config["debate"].get("rounds", 10) * 2 # Load plenty
        )
        # DeterministicAgent doesn't use MemoryStore in the standard way?
        # It doesn't use backend either.
        # But 'memory' variable is needed for logging/eval?
        # DeterministicAgent doesn't expose .memory property by default.
        # Orchestrator uses 'memory' variable in line 241 (seeding).
        # We need to decide: does DeterministicAgent use the 'store' created above?
        # Likely NO. It uses its own logic.
        # So we should probably disable 'memory' seeding for it or make it compatible.
        # For now, let's keep 'memory' as the unconnected store created above (lines 207-220),
        # so logging works but Agent ignores it.
        pass
        
    else:
        subject = SubjectAgent(backend, memory, config["debate"]["subject_persona"], name="Subject")
    
    # Instantiate Counter Agent
    counter_agent_type = config["debate"].get("counter_agent_type", "standard")
    counter_side = "con" # Default, or read from config if needed
    
    if "deterministic" in counter_agent_type:
        from src.agents.deterministic_agent import DeterministicAgent
        logger.info("Instantiating DeterministicAgent as Counter...")
        counter = DeterministicAgent(
            name="Counter", 
            topic=topic_name, 
            side=counter_side, 
            seed=config["debate"].get("seed", 42) + 1, # Offset seed to avoid syncing with subject if both deterministic
            limit=config["debate"].get("rounds", 10) * 2
        )
    else:
        counter = CounterAgent(backend, side=counter_side, topic=topic_name, config=config)
    extractor = ArgumentExtractionAgent(backend, topic_name, config=config)
    evaluator = EvaluationAgent(evaluator_backend)

    # 3.5 Seed Memory
    seed_stance = config["debate"].get("seed_stance", "none")
    if seed_stance != "none":
        logger.info(f"Seeding memory with stance: {seed_stance}")
        
        # Check data source configuration
        data_config = config.get("data", {})
        data_source = data_config.get("source", "json_files")  # "json_files" or "unified_json"
        
        if data_source == "unified_json":
            # Load from unified JSON file
            from src.utils.json_data_loader import load_arguments_from_json
            
            json_path = data_config.get("json_path", "data/Argument-Quality-Ranking-30k/arguments.json")
            
            args_to_load = []
            # Seeding Configuration
            seeding_limit = config["debate"].get("seeding_limit", 50) # Default to 50
            seeding_strategy = config["debate"].get("seeding_strategy", "quality_max")
            num_bins = config["debate"].get("seeding_num_bins", 5) # Default to 5 bins
            quality_lower_bound = config["debate"].get("seeding_quality_lower_bound", None) # For "random" strategy
            quality_upper_bound = config["debate"].get("seeding_quality_upper_bound", None) # For "random" strategy

            if seed_stance in [POLARITY_AFFIRMATIVE, "pro", "neutral"]:
                try:
                    pro_args = load_arguments_from_json(
                        json_path=json_path,
                        topic=topic_name,
                        side=POLARITY_AFFIRMATIVE,
                        limit=seeding_limit if seed_stance in [POLARITY_AFFIRMATIVE, "pro"] else seeding_limit // 2,
                        strategy=seeding_strategy,
                        num_bins=num_bins,
                        quality_lower_bound=quality_lower_bound,
                        quality_upper_bound=quality_upper_bound,
                        random_seed=seeding_random_seed
                    )
                    args_to_load.extend(pro_args)
                except Exception as e:
                    logger.error(f"Failed to load affirmative arguments from unified JSON: {e}")
            
            if seed_stance in [POLARITY_NEGATIVE, "con", "neutral"]:
                try:
                    con_args = load_arguments_from_json(
                        json_path=json_path,
                        topic=topic_name,
                        side=POLARITY_NEGATIVE,
                        limit=seeding_limit if seed_stance in [POLARITY_NEGATIVE, "con"] else seeding_limit // 2,
                        strategy=seeding_strategy,
                        num_bins=num_bins,
                        quality_lower_bound=quality_lower_bound,
                        quality_upper_bound=quality_upper_bound,
                        random_seed=None if seeding_random_seed is None else seeding_random_seed + 1
                    )
                    args_to_load.extend(con_args)
                except Exception as e:
                    logger.error(f"Failed to load negative arguments from unified JSON: {e}")
            
            # Add arguments to memory
            if isinstance(memory, AgenticMemoryStore):
                memory.seed_arguments(args_to_load)
                # Calculate and print pairwise cosine similarity statistics
                # Use the existing credence_manager from memory store
                print_seeding_similarity_stats(memory, memory.credence_manager) # Comment here to disable pairwise similarity printing
            else:
                # For regular MemoryStore, filter by similarity and fill up if needed
                # Use the existing credence_manager from memory store
                credence_manager = memory.credence_manager
                
                # Get existing arguments in memory (should be empty at seeding, but check anyway)
                existing_args = list(memory.arguments.values())
                
                # Filter arguments by similarity threshold
                # Only keep arguments that are sufficiently different from existing ones
                accepted_args = []
                rejected_args = []
                
                for candidate in args_to_load:
                    # Check similarity against existing memory
                    similar_to_existing = credence_manager.find_similar_arguments(candidate, existing_args)
                    
                    # Check similarity against already accepted arguments in this batch
                    similar_to_accepted = credence_manager.find_similar_arguments(candidate, accepted_args)
                    
                    if not similar_to_existing and not similar_to_accepted:
                        # Sufficiently different - accept it
                        accepted_args.append(candidate)
                        existing_args.append(candidate)  # Add to existing for next comparisons
                    else:
                        # Too similar - reject it
                        rejected_args.append(candidate)
                
                # Calculate target counts for each side based on seed_stance
                if seed_stance in [POLARITY_AFFIRMATIVE, "pro"]:
                    pro_target = seeding_limit
                    con_target = 0
                elif seed_stance in [POLARITY_NEGATIVE, "con"]:
                    pro_target = 0
                    con_target = seeding_limit
                else:  # neutral
                    pro_target = seeding_limit // 2
                    con_target = seeding_limit // 2
                
                # Calculate how many we currently have for each side
                pro_count = sum(1 for arg in accepted_args if arg.polarity == POLARITY_AFFIRMATIVE)
                con_count = sum(1 for arg in accepted_args if arg.polarity == POLARITY_NEGATIVE)
                
                # Track which arguments we've already tried (by claim text to avoid duplicates)
                tried_claims = {arg.claim for arg in args_to_load}
                
                # Fill up pro arguments if needed
                if pro_target > 0 and pro_count < pro_target:
                    attempts = 0
                    max_attempts = 5  # Limit retry attempts
                    while pro_count < pro_target and attempts < max_attempts:
                        attempts += 1
                        # Load more arguments (with higher limit to have more candidates)
                        additional_limit = (pro_target - pro_count) * 3  # Load 3x what we need
                        try:
                            additional_pro = load_arguments_from_json(
                                json_path=json_path,
                                topic=topic_name,
                                side=POLARITY_AFFIRMATIVE,
                                limit=additional_limit,
                                strategy=seeding_strategy,
                                num_bins=num_bins,
                                quality_lower_bound=quality_lower_bound,
                                quality_upper_bound=quality_upper_bound
                            )
                            
                            # Filter out already tried arguments
                            new_candidates = [arg for arg in additional_pro if arg.claim not in tried_claims]
                            
                            # Filter by similarity
                            for candidate in new_candidates:
                                if pro_count >= pro_target:
                                    break
                                
                                tried_claims.add(candidate.claim)
                                
                                similar_to_existing = credence_manager.find_similar_arguments(candidate, existing_args)
                                similar_to_accepted = credence_manager.find_similar_arguments(candidate, accepted_args)
                                
                                if not similar_to_existing and not similar_to_accepted:
                                    accepted_args.append(candidate)
                                    existing_args.append(candidate)
                                    pro_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to load additional pro arguments (attempt {attempts}): {e}")
                            break
                
                # Fill up con arguments if needed
                if con_target > 0 and con_count < con_target:
                    attempts = 0
                    max_attempts = 5  # Limit retry attempts
                    while con_count < con_target and attempts < max_attempts:
                        attempts += 1
                        # Load more arguments (with higher limit to have more candidates)
                        additional_limit = (con_target - con_count) * 3  # Load 3x what we need
                        try:
                            additional_con = load_arguments_from_json(
                                json_path=json_path,
                                topic=topic_name,
                                side=POLARITY_NEGATIVE,
                                limit=additional_limit,
                                strategy=seeding_strategy,
                                num_bins=num_bins,
                                quality_lower_bound=quality_lower_bound,
                                quality_upper_bound=quality_upper_bound
                            )
                            
                            # Filter out already tried arguments
                            new_candidates = [arg for arg in additional_con if arg.claim not in tried_claims]
                            
                            # Filter by similarity
                            for candidate in new_candidates:
                                if con_count >= con_target:
                                    break
                                
                                tried_claims.add(candidate.claim)
                                
                                similar_to_existing = credence_manager.find_similar_arguments(candidate, existing_args)
                                similar_to_accepted = credence_manager.find_similar_arguments(candidate, accepted_args)
                                
                                if not similar_to_existing and not similar_to_accepted:
                                    accepted_args.append(candidate)
                                    existing_args.append(candidate)
                                    con_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to load additional con arguments (attempt {attempts}): {e}")
                            break
                
                # Add all accepted arguments to memory
                for record in accepted_args:
                    record.source_role = "seed"
                    memory.add_argument(record)
                
                # Make all arguments in memory credence relevant (as requested for seeding)
                all_memory_args = list(memory.arguments.values())
                for arg in all_memory_args:
                    if hasattr(arg, 'credence_relevant'):
                        arg.credence_relevant = True
                    elif isinstance(arg, dict):
                        arg['credence_relevant'] = True
                
                logger.info(f"Seeded {len(accepted_args)} arguments (pro: {pro_count}, con: {con_count}) after similarity filtering")
                
                # Calculate and print pairwise cosine similarity statistics
                print_seeding_similarity_stats(memory, credence_manager) # Comment here to disable pairwise similarity printing
        else:
            # Original JSON files approach (backward compatibility)
            topic_dir = os.path.join("data/topics", topic_name)
            
            args_to_load = []
            if seed_stance in [POLARITY_AFFIRMATIVE, "pro", "neutral"]:
                try:
                    with open(os.path.join(topic_dir, "pro.json"), "r") as f:
                        args_to_load.extend(json.load(f))
                except FileNotFoundError:
                    logger.error("pro.json not found for seeding")

            if seed_stance in [POLARITY_NEGATIVE, "con", "neutral"]:
                try:
                    with open(os.path.join(topic_dir, "con.json"), "r") as f:
                        args_to_load.extend(json.load(f))
                except FileNotFoundError:
                    logger.error("con.json not found for seeding")
            
            records = []
            for arg_data in args_to_load:
                # Ensure required fields
                arg_data["topic_id"] = topic_name
                arg_data["source_role"] = "seed"
                record = ArgumentRecord.from_dict({
                    **arg_data, 
                    "timestamp": arg_data.get("timestamp", "2024-01-01T00:00:00") # Dummy timestamp
                })
                records.append(record)
            
            # Add arguments to memory
            if isinstance(memory, AgenticMemoryStore):
                memory.seed_arguments(records)
                # Calculate and print pairwise cosine similarity statistics
                # Use the existing credence_manager from memory store
                print_seeding_similarity_stats(memory, memory.credence_manager) # Comment here to disable pairwise similarity printing
            else:
                # For regular MemoryStore, filter by similarity and fill up if needed
                # Use the existing credence_manager from memory store
                credence_manager = memory.credence_manager
                
                # Get existing arguments in memory (should be empty at seeding, but check anyway)
                existing_args = list(memory.arguments.values())
                
                # Filter arguments by similarity threshold
                # Only keep arguments that are sufficiently different from existing ones
                accepted_args = []
                
                for candidate in records:
                    # Check similarity against existing memory
                    similar_to_existing = credence_manager.find_similar_arguments(candidate, existing_args)
                    
                    # Check similarity against already accepted arguments in this batch
                    similar_to_accepted = credence_manager.find_similar_arguments(candidate, accepted_args)
                    
                    if not similar_to_existing and not similar_to_accepted:
                        # Sufficiently different - accept it
                        accepted_args.append(candidate)
                        existing_args.append(candidate)  # Add to existing for next comparisons
                
                # Add all accepted arguments to memory
                for record in accepted_args:
                    record.source_role = "seed"
                    memory.add_argument(record)
                
                # Make all arguments in memory credence relevant (as requested for seeding)
                all_memory_args = list(memory.arguments.values())
                for arg in all_memory_args:
                    if hasattr(arg, 'credence_relevant'):
                        arg.credence_relevant = True
                    elif isinstance(arg, dict):
                        arg['credence_relevant'] = True
                
                logger.info(f"Seeded {len(accepted_args)} arguments after similarity filtering (from {len(records)} loaded)")
                
                # Calculate and print pairwise cosine similarity statistics
                print_seeding_similarity_stats(memory, credence_manager) # Comment here to disable pairwise similarity printing


    # 4. Pre-Debate Evaluation
    initial_memory_snapshot = memory.get_all_arguments() # Capture for comparison
    print("\n" + "="*50)
    print("PRE-DEBATE CONFIGURATION")
    print("="*50)
    print(f"Topic: {topic_name}")
    print(f"Storage Mode: {storage_mode}")
    if storage_mode == "worldview":
        print(f"Worldview Openness: {worldview_openness}")
    print(f"Retrieval Strategy: {retrieval_strategy}")
    # Check Counter Stance
    counter_stance = counter.get_stance()
    counter_side_str = f"{counter.side.upper()} ({POLARITY_DISPLAY.get(counter.side, 'Unknown')})"
    print(f"Counter Agent Side: {counter_side_str} (Stance Value: {counter_stance})")
    
    # Check Subject Stance (Seeded)
    seeded_stance = subject.get_current_stance()
    print(f"Subject Agent Seeded Stance (Calculated): {seeded_stance:.2f}")
    print("="*50 + "\n")
    
    logger.log_event("pre_eval", {
        "seeded_voter_stance": seeded_stance,
        "counter_stance": counter_stance
    })

    # 5. Run Debate
    # Use SimulationRunner (Consolidated Engine)
    runner = SimulationRunner(
        agents=[subject, counter], 
        rounds=config["debate"]["rounds"], 
        extractor=extractor, 
        evaluator=evaluator,
        output_dir=logger.run_dir,
        speaker_order=config.get("simulation", {}).get(
            "speaker_order",
            "pro-first",
        ),
    )
    
    # Execute Simulation
    sim_results = runner.run()
    
    transcript = sim_results["transcript"]
    stance_history = sim_results["stance_history"]
    
    # 5.1 Extract Opening Statement for "Initial" context
    opening_statement = "No opening statement found."
    if transcript and len(transcript) > 0:
        # Assuming first message is from Subject (Assistant)
        if transcript[0]['role'] == 'assistant':
            opening_statement = transcript[0]['content']
    
    # Capture Initial Stance (from history or agent)
    # SimulationRunner records round 0 in stance_history[0]
    initial_stance = subject.get_current_stance() # Fallback to current if history implies something else, but simulated history is safer
    if stance_history:
        initial_stance = stance_history[0].get(subject.name, initial_stance)
        
    logger.log_event("initial_stance_post_opening", {"stance": initial_stance})
    
    # 4.5 Initial Qualitative Evaluation (Post-Hoc calculation using Opening Statement)
    # We use the opening statement + initial memory context
    initial_memory_context = str(memory.get_all_arguments())
    initial_eval_result = evaluator.evaluate_stance(
        f"Persona: {config['debate']['subject_persona']}\nOpening Statement: {opening_statement}", 
        initial_memory_context,
        topic=topic_name,
        agent_name="Subject"
    )
    print(f"Evaluator Estimated Initial Stance (LLM-Based): {initial_eval_result.get('stance_score')}")
    logger.log_event("pre_eval_qualitative", {
        "initial_qualitative_stance": initial_eval_result.get("stance_score")
    })

    # 5.2 (Simulation already ran)

    # 6. Post-Debate Evaluation
    logger.info("Running post-debate evaluation...")
    final_stance = subject.get_current_stance()
    
    # Convert transcript to string
    role_map = {"user": "Counter Agent", "assistant": "Subject Agent"}
    transcript_text = "\n".join([f"{role_map.get(msg['role'], msg['role'])}: {msg['content']}" for msg in transcript])
    memory_snapshot = str(memory.get_all_arguments())
    
    # Extract only the final statement from the Subject Agent for qualitative evaluation
    final_voter_msg = "No final statement found."
    for msg in reversed(transcript):
        if msg['role'] == "assistant": # Subject Agent
            final_voter_msg = msg['content']
            break
    
    # Get the stance instruction from belief engine for consistency check
    stance_instruction = None
    if hasattr(memory, 'belief_engine') and memory.belief_engine:
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

    # Get final probability
    final_prob = memory.get_bayesian_probability() if hasattr(memory, "get_bayesian_probability") else None
    prob_str = f" (Prob: {final_prob:.4f})" if final_prob is not None else ""

    logger.log_event("post_eval", {
        "final_calculated_stance": final_stance,
        "final_bayesian_probability": final_prob,
        "evaluator_estimated_stance": eval_result.get("stance_score"),
        "consistency": consistency_result,
        "summary": summary
    })

    logger.info(f"Final Quantitative Stance (Calculated): {final_stance}{prob_str}")
    logger.info(f"Final Qualitative Stance (LLM-Based): {eval_result.get('stance_score')}")
    
    # Calculate Memory Changes
    final_memory = memory.get_all_arguments()
    initial_claims = {arg['claim'] for arg in initial_memory_snapshot}
    
    new_arguments = []
    
    for arg in final_memory:
        if arg['claim'] not in initial_claims:
            new_arguments.append(arg)

    memory_summary = f"\n\n## Memory Evolution\n"
    memory_summary += f"- **Initial Arguments**: {len(initial_memory_snapshot)}\n"
    memory_summary += f"- **Final Arguments**: {len(final_memory)}\n"
    memory_summary += f"- **New Arguments Acquired**: {len(new_arguments)}\n\n"
    
    if new_arguments:
        memory_summary += "### Newly Acquired Arguments:\n"
        for arg in new_arguments:
            memory_summary += f"- [{arg['polarity'].upper()}] {arg['claim']} (Strength: {arg['support_strength']:.2f})\n"

    # Define descriptions for storage modes
    mode_descriptions = {
        "episodic": "Stores every argument as a distinct record.",
        "worldview": "Stores arguments with an additional worldview-consistency check.",
        "agentic": "Uses the optional Chroma-backed experimental memory store.",
    }
    mode_desc = mode_descriptions.get(config["memory"]["storage_mode"], "Unknown mode")

    stance_method_label = config["memory"].get("stance_method", "linear")
    belief_engine_type = config["memory"].get("belief_engine_type", None)
    if belief_engine_type in ("agentic", "llm"):
        stance_method_label = "agentic"

    full_summary = f"""# Experiment Summary: {config["debate"]["topic"]}

## Experiment Configuration
- **Run ID**: {logger.run_id}
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Rounds**: {config["debate"]["rounds"]}
- **Seed Stance**: {config["debate"].get("seed_stance", "none")}
- **Subject Persona**: {config["debate"]["subject_persona"]}
- **Memory Decay Factor**: {config["memory"].get("decay_factor", 1.0)}
- **Retrieval Strategy**: {config["memory"].get("retrieval_strategy", "mixed")}
- **Storage Mode**: {config["memory"]["storage_mode"]}
  - *{mode_desc}*
- **Worldview Openness**: {config["memory"].get("worldview_openness", "balanced")} (Only for 'worldview' mode)
- **Stance Method**: {stance_method_label}
- **Update Sensitivity**: {config["memory"].get("update_sensitivity", "N/A")}
- **Anchor Sensitivity**: {config["memory"].get("anchor_sensitivity", "N/A")}
- **Confirmation Bias**: {config["memory"].get("confirmation_bias", "0.0")}

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

    logger.log_summary(full_summary)
    
    print("\n" + "="*50)
    print("DEBATE SUMMARY")
    print("="*50)
    print(summary)
    print("="*50 + "\n")
    
    logger.info("Experiment complete.")

    if isinstance(memory, AgenticMemoryStore): # avoid orphaned process
        memory.close()

    if return_results:
        return {
            "run_id": logger.run_id,
            "storage_mode": storage_mode,
            "worldview_openness": worldview_openness if storage_mode == "worldview" else "N/A",
            "seeded_stance": seeded_stance,
            "initial_stance": initial_stance,
            "final_stance": final_stance,
            "stance_shift": final_stance - initial_stance,
            "final_memory_count": len(final_memory),
            "new_arguments_count": len(new_arguments),
            "evaluator_stance": eval_result.get("stance_score"),
            "replacements": getattr(runner, "replacements", runner.argument_updates),
            "stance_history": stance_history,
            "transcript": transcript,
            "log_dir": logger.run_dir
        }

if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"\n\n[FATAL ERROR] Experiment failed: {e}")
        print("Exiting immediately due to error.")
        sys.exit(1)
