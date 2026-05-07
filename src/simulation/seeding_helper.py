import os
import json
import logging
import random
import math
import hashlib
from typing import Dict, Any, List, Optional
from collections import defaultdict

from src.memory.memory_store import MemoryStore, ArgumentRecord
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_MAPPING
from src.utils.json_data_loader import load_arguments_from_json

logger = logging.getLogger(__name__)


def _stable_seed(base_seed: Optional[int], *parts: str) -> Optional[int]:
    """Derive a deterministic per-topic/per-side seed from a campaign seed."""
    if base_seed is None:
        return None

    seed_material = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)

def print_seeding_similarity_stats(memory, credence_manager):
    """
    Calculate and print pairwise cosine similarity statistics for seeded arguments.
    """
    all_memory_args = list(memory.arguments.values())
    
    if len(all_memory_args) > 1:
        embeddings = []
        for arg in all_memory_args:
            embedding = credence_manager._get_embedding(arg.claim)
            embeddings.append(embedding)
        
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

def seed_review_artifactory(memory: MemoryStore, topic: str, side: str, config: Dict[str, Any]):
    """
    Seeds an agent's memory with arguments for a specific side of a topic.
    Includes pairwise similarity deduplication, dynamic tracking, and retry logic.
    """
    if not side or side == "none":
        return

    logger.info(f"Seeding memory for topic '{topic}' with side '{side}'")
    
    # Check data source configuration
    data_config = config.get("data", {})
    data_source = data_config.get("source", "json_files")
    
    debate_config = config.get("debate", {})
    seeding_limit = debate_config.get("seeding_limit", 10)
    seeding_strategy = debate_config.get("seeding_strategy", "quality_max")
    seed_dedup = debate_config.get("seed_dedup", True)
    num_bins = debate_config.get("seeding_num_bins", 5)
    quality_lower_bound = debate_config.get("seeding_quality_lower_bound", None)
    quality_upper_bound = debate_config.get("seeding_quality_upper_bound", None)
    base_random_seed = debate_config.get("seeding_random_seed", None)
    side_random_seed = _stable_seed(base_random_seed, topic, side)

    if data_source == "unified_json":
        json_path = data_config.get("json_path", "data/Argument-Quality-Ranking-30k/arguments.json")
        args_to_load = []
        
        if side in [POLARITY_AFFIRMATIVE, "pro", "neutral"]:
            try:
                pro_args = load_arguments_from_json(
                    json_path=json_path,
                    topic=topic,
                    side=POLARITY_AFFIRMATIVE,
                    limit=seeding_limit if side in [POLARITY_AFFIRMATIVE, "pro"] else seeding_limit // 2,
                    strategy=seeding_strategy,
                    num_bins=num_bins,
                    quality_lower_bound=quality_lower_bound,
                    quality_upper_bound=quality_upper_bound,
                    random_seed=side_random_seed,
                )
                args_to_load.extend(pro_args)
            except Exception as e:
                logger.error(f"Failed to load affirmative arguments: {e}")
        
        if side in [POLARITY_NEGATIVE, "con", "neutral"]:
            try:
                con_args = load_arguments_from_json(
                    json_path=json_path,
                    topic=topic,
                    side=POLARITY_NEGATIVE,
                    limit=seeding_limit if side in [POLARITY_NEGATIVE, "con"] else seeding_limit // 2,
                    strategy=seeding_strategy,
                    num_bins=num_bins,
                    quality_lower_bound=quality_lower_bound,
                    quality_upper_bound=quality_upper_bound,
                    random_seed=side_random_seed,
                )
                args_to_load.extend(con_args)
            except Exception as e:
                logger.error(f"Failed to load negative arguments: {e}")
        
        # Check Agentic Memory
        from src.memory.agentic_memory import AgenticMemoryStore
        if isinstance(memory, AgenticMemoryStore):
            memory.seed_arguments(args_to_load)
            print_seeding_similarity_stats(memory, memory.credence_manager)
        else:
            credence_manager = memory.credence_manager
            existing_args = list(memory.arguments.values())
            accepted_args = []
            rejected_args = []
            
            if not seed_dedup:
                accepted_args = list(args_to_load)
            else:
                for candidate in args_to_load:
                    similar_to_existing = credence_manager.find_similar_arguments(candidate, existing_args)
                    similar_to_accepted = credence_manager.find_similar_arguments(candidate, accepted_args)

                    if not similar_to_existing and not similar_to_accepted:
                        accepted_args.append(candidate)
                        existing_args.append(candidate)
                    else:
                        rejected_args.append(candidate)

            # Target Counts
            if side in [POLARITY_AFFIRMATIVE, "pro"]:
                pro_target = seeding_limit
                con_target = 0
            elif side in [POLARITY_NEGATIVE, "con"]:
                pro_target = 0
                con_target = seeding_limit
            else:
                pro_target = seeding_limit // 2
                con_target = seeding_limit // 2

            pro_count = sum(1 for arg in accepted_args if arg.polarity == POLARITY_AFFIRMATIVE)
            con_count = sum(1 for arg in accepted_args if arg.polarity == POLARITY_NEGATIVE)
            tried_claims = {arg.claim for arg in args_to_load}

            # Fill up pro arguments if needed
            if seed_dedup and pro_target > 0 and pro_count < pro_target:
                attempts = 0
                while pro_count < pro_target and attempts < 5:
                    attempts += 1
                    try:
                        additional_pro = load_arguments_from_json(
                            json_path=json_path, topic=topic, side=POLARITY_AFFIRMATIVE,
                            limit=(pro_target - pro_count) * 3, strategy=seeding_strategy,
                            num_bins=num_bins, quality_lower_bound=quality_lower_bound, quality_upper_bound=quality_upper_bound,
                            random_seed=side_random_seed,
                        )
                        for candidate in [arg for arg in additional_pro if arg.claim not in tried_claims]:
                            if pro_count >= pro_target: break
                            tried_claims.add(candidate.claim)
                            if not credence_manager.find_similar_arguments(candidate, existing_args) and not credence_manager.find_similar_arguments(candidate, accepted_args):
                                accepted_args.append(candidate)
                                existing_args.append(candidate)
                                pro_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load additional pro arguments: {e}")
                        break

            # Fill up con arguments if needed
            if seed_dedup and con_target > 0 and con_count < con_target:
                attempts = 0
                while con_count < con_target and attempts < 5:
                    attempts += 1
                    try:
                        additional_con = load_arguments_from_json(
                            json_path=json_path, topic=topic, side=POLARITY_NEGATIVE,
                            limit=(con_target - con_count) * 3, strategy=seeding_strategy,
                            num_bins=num_bins, quality_lower_bound=quality_lower_bound, quality_upper_bound=quality_upper_bound,
                            random_seed=side_random_seed,
                        )
                        for candidate in [arg for arg in additional_con if arg.claim not in tried_claims]:
                            if con_count >= con_target: break
                            tried_claims.add(candidate.claim)
                            if not credence_manager.find_similar_arguments(candidate, existing_args) and not credence_manager.find_similar_arguments(candidate, accepted_args):
                                accepted_args.append(candidate)
                                existing_args.append(candidate)
                                con_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load additional con arguments: {e}")
                        break

            # Add all accepted arguments to memory
            for record in accepted_args:
                record.source_role = "seed"
                memory.add_argument(record)

            # Make credence relevant
            for arg in memory.arguments.values():
                if hasattr(arg, 'credence_relevant'):
                    arg.credence_relevant = True
                elif isinstance(arg, dict):
                    arg['credence_relevant'] = True

            logger.info(f"Seeded {len(accepted_args)} arguments (pro: {pro_count}, con: {con_count}, seed_dedup={seed_dedup})")
            print_seeding_similarity_stats(memory, credence_manager)
    else:
        # Legacy JSON approach
        target_side = POLARITY_MAPPING.get(side.lower(), side)
        file_side = "pro" if target_side == POLARITY_AFFIRMATIVE else "con"
        file_path = os.path.join("data/topics", topic, f"{file_side}.json")
        try:
            with open(file_path, "r") as f:
                raw_data = json.load(f)
                args_to_load_dicts = raw_data[:seeding_limit * 3]
                records = []
                for item in args_to_load_dicts:
                    item["topic_id"] = topic
                    item["source_role"] = "seed"
                    if "timestamp" not in item: item["timestamp"] = "2024-01-01T00:00:00"
                    records.append(ArgumentRecord.from_dict(item))

                from src.memory.agentic_memory import AgenticMemoryStore
                if isinstance(memory, AgenticMemoryStore):
                    memory.seed_arguments(records)
                    print_seeding_similarity_stats(memory, memory.credence_manager)
                else:
                    credence_manager = memory.credence_manager
                    existing_args = list(memory.arguments.values())
                    accepted_args = []
                    for candidate in records:
                        if len(accepted_args) >= seeding_limit: break
                        if not seed_dedup or (
                            not credence_manager.find_similar_arguments(candidate, existing_args)
                            and not credence_manager.find_similar_arguments(candidate, accepted_args)
                        ):
                            accepted_args.append(candidate)
                            existing_args.append(candidate)

                    for record in accepted_args:
                        record.source_role = "seed"
                        memory.add_argument(record)

                    for arg in memory.arguments.values():
                        if hasattr(arg, 'credence_relevant'): arg.credence_relevant = True
                        elif isinstance(arg, dict): arg['credence_relevant'] = True
                    print_seeding_similarity_stats(memory, credence_manager)
        except FileNotFoundError:
            logger.warning(f"No arguments file found for topic {topic} side {file_side}: {file_path}")
        except Exception as e:
            logger.error(f"Failed to seed memory: {e}", exc_info=True)
            raise
    try:
        logger.info(f"Agent starting stance after seeding: {memory.get_stance():.4f}")
    except:
        pass


def get_seed_arguments(topic: str, side: str, limit: int, config: Dict[str, Any]) -> List[str]:
    """
    Returns a list of argument claim strings for seeding a vanilla agent.
    Used to provide the same initial information as memory-based agents.
    """
    target_side = POLARITY_MAPPING.get(side.lower(), None)
    if not target_side:
        return []
    
    data_config = config.get("data", {})
    data_source = data_config.get("source", "json_files")
    
    if data_source == "unified_json":
        json_path = data_config.get("json_path", "data/Argument-Quality-Ranking-30k/arguments.json")
        
        try:
            args = load_arguments_from_json(
                json_path=json_path,
                topic=topic,
                side=target_side,
                limit=limit,
                strategy="random"
            )
            return [arg.claim for arg in args]
        except Exception as e:
            logger.warning(f"Failed to load seed arguments: {e}")
            return []
    
    return []

def seed_agents_collaboratively(agents: List[Any], agent_configs: List[Dict], topic: str, config: Dict[str, Any]):
    """
    Seeds multiple agents simultaneously to ensure balanced score distributions (score_matched).
    Finds the intersection of score histograms between Pro and Con datasets and samples
    to ensure both sides have the same quality distribution.
    """
    logger.info(f"Collaborative Seeding: 'score_matched' for topic '{topic}'")
    
    debate_config = config.get("debate", {})
    seeding_limit = debate_config.get("seeding_limit", 10)
    
    # Identify sides for agents
    side_agents = defaultdict(list)
    
    for agent, conf in zip(agents, agent_configs):
        if not hasattr(agent, 'memory'):
            continue
            
        # Determine side (Subject usually has 'seed_stance' or target_initial_stance)
        # Counter agent usually has 'side'
        # normalized to 'affirmative' / 'negative'
        raw_side = conf.get("seed_stance") or conf.get("side")
        if not raw_side:
             continue
             
        side = POLARITY_MAPPING.get(raw_side.lower(), None)
        if side:
            side_agents[side].append(agent)
            
    if not side_agents[POLARITY_AFFIRMATIVE] or not side_agents[POLARITY_NEGATIVE]:
        logger.warning("Collaborative seeding requires at least one affirmative and one negative agent. Falling back to individual seeding.")
        # Fallback to individual
        for agent, conf in zip(agents, agent_configs):
            if hasattr(agent, 'memory'):
                start_stance = conf.get("seed_stance") or conf.get("side")
                if start_stance:
                     seed_review_artifactory(agent.memory, topic, start_stance, config)
        return

    # Load ALL arguments for both sides
    # We use a large limit to get everything (e.g. 5000)
    json_path = config.get("data", {}).get("json_path", "data/Argument-Quality-Ranking-30k/arguments.json")
    
    # We use "random" strategy with large limit to get a raw sample without pre-sorting
    pool_aff = load_arguments_from_json(json_path, topic, side=POLARITY_AFFIRMATIVE, limit=5000, strategy="random")
    pool_neg = load_arguments_from_json(json_path, topic, side=POLARITY_NEGATIVE, limit=5000, strategy="random")
    
    if len(pool_aff) < seeding_limit or len(pool_neg) < seeding_limit:
        logger.warning(f"Not enough arguments for collaborative seeding (Aff: {len(pool_aff)}, Neg: {len(pool_neg)}). Limit: {seeding_limit}.")
        return

    # Create Bins
    num_bins = 10
    # We don't need explicit bin edges if we just map floor(val * 10)
    
    def bin_args(args):
        binned = defaultdict(list)
        for arg in args:
            idx = min(int(arg.support_strength * num_bins), num_bins - 1)
            binned[idx].append(arg)
        return binned

    binned_aff = bin_args(pool_aff)
    binned_neg = bin_args(pool_neg)
    
    # Calculate Intersection (Capacity) of the underlying datasets
    intersection_caps = {}
    total_cap = 0
    for i in range(num_bins):
        cap = min(len(binned_aff[i]), len(binned_neg[i]))
        intersection_caps[i] = cap
        total_cap += cap
        
    logger.info(f"Score Matched Intersection Total Capacity: {total_cap} arguments.")
    
    if total_cap < seeding_limit:
        logger.warning("Total intersection capacity is less than seeding limit. Filling with best effort.")
        # Logic allows picking up to capacity, so it will just pick what it can.
    
    # Determine target counts per bin
    # Probability distribution of the INTERSECTION
    bin_probs = {i: intersection_caps[i] / total_cap for i in range(num_bins)} if total_cap > 0 else {i:0 for i in range(num_bins)}
    
    # Distribute seeding_limit according to probs to determine how many to pick per bin
    target_counts = {i: 0 for i in range(num_bins)}
    remaining = seeding_limit
    
    # First pass: floor
    for i in range(num_bins):
        count = int(bin_probs[i] * seeding_limit)
        target_counts[i] = count
        remaining -= count
        
    # Second pass: distribute remainder to bins with highest fractional part
    # Sort by fractional part Descending
    sorted_bin_indices = sorted(bin_probs.keys(), key=lambda k: bin_probs[k] * seeding_limit - int(bin_probs[k] * seeding_limit), reverse=True)
    
    for i in range(remaining):
        # Determine which bin to add 1 to
        # Use modulo if remaining > num_bins (unlikely)
        idx = sorted_bin_indices[i % len(sorted_bin_indices)]
        target_counts[idx] += 1
        
    logger.info(f"Seeding Targets per Bin: {target_counts}")
    
    # Sample and Assign
    final_aff = []
    final_neg = []
    
    for i in range(num_bins):
        count = target_counts[i]
        # Safety check if count > capacity (should not happen if logic is right, but safe due to rounding)
        # Actually if we requested more than capacity (e.g. limit > capacity), this caps it.
        count = min(count, intersection_caps[i])
        
        if count > 0:
            final_aff.extend(random.sample(binned_aff[i], count))
            final_neg.extend(random.sample(binned_neg[i], count))
            
    # Add to agents
    # For Affirmative Agents
    for agent in side_agents[POLARITY_AFFIRMATIVE]:
        logger.info(f"Seeding {agent.name} (Aff) with {len(final_aff)} balanced arguments.")
        for arg in final_aff:
            # We add argument. Note: backend=None because we trust the seed.
            # However, we must ensure it's treated as a seed.
            arg.source_role = "seed"
            agent.memory.add_argument(arg, backend=None)
        # Recalculate stance
        try:
             logger.info(f"Agent {agent.name} starting stance: {agent.memory.get_stance():.4f}")
        except: pass
            
    # For Negative Agents
    for agent in side_agents[POLARITY_NEGATIVE]:
        logger.info(f"Seeding {agent.name} (Neg) with {len(final_neg)} balanced arguments.")
        for arg in final_neg:
            arg.source_role = "seed"
            agent.memory.add_argument(arg, backend=None)
        try:
             logger.info(f"Agent {agent.name} starting stance: {agent.memory.get_stance():.4f}")
        except: pass

    logger.info(f"Collaborative seeding complete.")
