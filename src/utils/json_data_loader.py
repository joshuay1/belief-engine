"""
Utility to load arguments from the unified JSON dataset.
"""

import json
import os
import random
from typing import List, Dict, Optional
from ..memory.argument_record import ArgumentRecord
from datetime import datetime
from ..constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE, POLARITY_MAPPING


def load_arguments_from_json(
    json_path: str,
    topic: str,
    side: Optional[str] = None,
    limit: Optional[int] = None,
    strategy: str = "quality_max", # "quality_max", "quality_balanced", or "random"
    num_bins: int = 5,  # Number of quality bins for "quality_balanced" strategy
    quality_lower_bound: Optional[float] = None,  # Lower bound for "random" strategy (filters by support_strength)
    quality_upper_bound: Optional[float] = None,  # Upper bound for "random" strategy (filters by support_strength)
    random_seed: Optional[int] = None
) -> List[ArgumentRecord]:
    """
    Load arguments from unified JSON file and convert to ArgumentRecord format.
    
    Args:
        json_path: Path to the unified JSON file (e.g., arguments.json)
        topic: Topic string to filter by (must match exactly as it appears in JSON)
        side: Optional filter by polarity ("pro", "con", or None for both)
        limit: Optional limit on number of arguments to return
        strategy: Seeding strategy - "quality_max" (top-k by support_strength), 
                 "quality_balanced" (balanced sampling across quality tiers), 
                 or "random" (random selection)
        num_bins: Number of quality bins to use for "quality_balanced" strategy (default: 5)
        quality_lower_bound: Lower bound for "random" strategy - only select arguments with 
                            support_strength >= this value (default: None, no lower bound)
        quality_upper_bound: Upper bound for "random" strategy - only select arguments with 
                            support_strength <= this value (default: None, no upper bound)
        random_seed: Optional deterministic RNG seed for stochastic seeding strategies
    
    Returns:
        List of ArgumentRecord objects
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        all_arguments = json.load(f)
    
    # Filter by topic (exact match or case-insensitive)
    filtered_args = []
    for arg_data in all_arguments:
        arg_topic = arg_data.get('topic', '')
        # Try exact match first, then case-insensitive
        if arg_topic == topic or arg_topic.lower() == topic.lower():
            # Filter by side if specified
            raw_polarity = arg_data.get('polarity', '')
            normalized_polarity = POLARITY_MAPPING.get(raw_polarity, raw_polarity)
            
            # Also normalize the requested side if it's in the mapping (it should be already, but to be safe)
            target_side = POLARITY_MAPPING.get(side, side) if side else None

            if target_side is None or normalized_polarity == target_side:
                filtered_args.append(arg_data)
    
    # Convert to ArgumentRecord objects
    records = []
    for arg_data in filtered_args:
        # Skip neutral arguments (current code only handles pro/con)
        raw_polarity = arg_data.get('polarity', '')
        if raw_polarity not in POLARITY_MAPPING:
            continue
            
        polarity = POLARITY_MAPPING[raw_polarity]
        
        record = ArgumentRecord(
            topic_id=topic,  # Use original topic ID from config
            claim=arg_data['claim'],
            polarity=polarity,
            source_role="seed",  # Will be overridden by caller if needed
            support_strength=float(arg_data.get('support_strength', 0.5)),
            evidence_snippets=arg_data.get('evidence_snippets', []),
            category_tags=arg_data.get('category_tags', []),
            timestamp=datetime.now()  # Use current time
        )
        records.append(record)
    
    # Apply Seeding Strategy and Limit
    if limit and limit > 0 and len(records) > limit:
        rng = random.Random(random_seed) if random_seed is not None else random
        if strategy == "quality_max":
            # Sort by support_strength descending (highest quality first)
            records.sort(key=lambda x: x.support_strength, reverse=True)
            records = records[:limit]
        elif strategy == "quality_balanced":
            # Sort by support_strength descending
            records.sort(key=lambda x: x.support_strength, reverse=True)
            
            # Divide into num_bins quality bins
            total = len(records)
            bin_size = total // num_bins
            
            # Create bins dynamically
            bins = []
            for i in range(num_bins):
                start_idx = i * bin_size
                if i == num_bins - 1:
                    # Last bin gets any remainder
                    bins.append(records[start_idx:])
                else:
                    bins.append(records[start_idx:start_idx + bin_size])
            
            # Sample evenly from each bin
            per_bin = limit // num_bins
            remainder = limit % num_bins
            
            selected = []
            
            # Sample from each bin
            for i, bin_records in enumerate(bins):
                if bin_records:
                    # Distribute remainder across first bins
                    sample_size = min(per_bin + (1 if remainder > 0 else 0), len(bin_records))
                    selected.extend(rng.sample(bin_records, sample_size))
                    remainder -= 1 if remainder > 0 else 0
            
            # If we still need more (due to uneven bin sizes), fill from remaining arguments
            if len(selected) < limit:
                remaining = [r for r in records if r not in selected]
                rng.shuffle(remaining)
                selected.extend(remaining[:limit - len(selected)])
            
            records = selected[:limit]
        elif strategy == "random":
            # Filter by quality bounds if specified
            if quality_lower_bound is not None or quality_upper_bound is not None:
                filtered_records = []
                for record in records:
                    strength = record.support_strength
                    # Check lower bound
                    if quality_lower_bound is not None and strength < quality_lower_bound:
                        continue
                    # Check upper bound
                    if quality_upper_bound is not None and strength > quality_upper_bound:
                        continue
                    filtered_records.append(record)
                records = filtered_records
            
            # Apply random selection
            rng.shuffle(records)
            records = records[:limit]
            
    return records


def load_arguments_for_counter(
    json_path: str,
    topic: str,
    side: str
) -> List[Dict]:
    """
    Load arguments for CounterAgent (returns dict format, not ArgumentRecord).
    
    Args:
        json_path: Path to the unified JSON file
        topic: Topic string to filter by (must match exactly as it appears in JSON)
        side: Polarity to filter by ("pro" or "con")
    
    Returns:
        List of dicts with 'claim', 'id', 'polarity', etc.
    """
    if not os.path.exists(json_path):
        print(f"Warning: JSON file not found: {json_path}")
        return []
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        all_arguments = json.load(f)
    
    # Filter by topic and side
    filtered_args = []
    for arg_data in all_arguments:
        arg_topic = arg_data.get('topic', '')
        arg_polarity = arg_data.get('polarity', '')
        normalized_polarity = POLARITY_MAPPING.get(arg_polarity, arg_polarity)
        target_side = POLARITY_MAPPING.get(side, side)
        
        # Match topic (exact or case-insensitive)
        if (arg_topic == topic or arg_topic.lower() == topic.lower()) and normalized_polarity == target_side:
            # Create dict in format expected by CounterAgent
            arg_dict = {
                'claim': arg_data['claim'],
                'polarity': arg_polarity,
                'support_strength': arg_data.get('support_strength', 0.5),
                'id': str(hash(arg_data['claim']))  # Generate ID from claim hash
            }
            filtered_args.append(arg_dict)
    
    return filtered_args


def get_available_topics(json_path: str) -> List[str]:
    """
    Get list of unique topics from the JSON file.
    
    Args:
        json_path: Path to the unified JSON file
    
    Returns:
        List of unique topic strings
    """
    if not os.path.exists(json_path):
        return []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        all_arguments = json.load(f)
    
    topics = set()
    for arg_data in all_arguments:
        topic = arg_data.get('topic', '')
        if topic:
            topics.add(topic)
    
    return sorted(list(topics))
