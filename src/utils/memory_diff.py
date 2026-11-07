from typing import Set, Dict, List
from src.memory.memory_store import MemoryStore
from src.memory.argument_record import ArgumentRecord
from src.constants import POLARITY_DISPLAY

def generate_memory_diff_md(
    agent_name: str,
    final_memory: MemoryStore,
    initial_arg_ids: Set[str],
    initial_stance: float,
    final_stance: float,
    topic: str
) -> str:
    """
    Generates a Markdown report comparing the agent's initial and final memory states.
    
    Args:
        agent_name: Name of the agent
        final_memory: The MemoryStore object at the end of simulation
        initial_arg_ids: Set of argument IDs that were present at start (seeded)
        initial_stance: Stance score at start
        final_stance: Stance score at end
        topic: The topic of debate
        
    Returns:
        A string containing the Markdown report
    """
    
    drift = final_stance - initial_stance
    drift_symbol = "+" if drift > 0 else ""
    
    # Categorize arguments
    all_args = final_memory.arguments.values()
    
    # Pre-existing (Seeded) - Active vs Inactive
    seeded_active = []
    seeded_inactive = []
    
    # New (Acquired) - Active vs Inactive
    new_active = []
    new_inactive = []
    
    def get_field(item, key, default=None):
        if isinstance(item, dict):
            return item.get(key, default)
        return getattr(item, key, default)

    def get_arg_id(item):
        arg_id = get_field(item, "id")
        if arg_id:
            return arg_id
        claim = get_field(item, "claim", "")
        return str(hash(claim)) if claim else ""

    for arg in all_args:
        arg_id = get_arg_id(arg)
        is_seeded = arg_id in initial_arg_ids
        is_active = bool(get_field(arg, "credence_relevant", False))
        
        if is_seeded:
            if is_active:
                seeded_active.append(arg)
            else:
                seeded_inactive.append(arg)
        else:
            if is_active:
                new_active.append(arg)
            else:
                new_inactive.append(arg)
                
    # Sort by strength (descending)
    seeded_active.sort(key=lambda x: get_field(x, "support_strength", 0.0), reverse=True)
    seeded_inactive.sort(key=lambda x: get_field(x, "support_strength", 0.0), reverse=True)
    new_active.sort(key=lambda x: get_field(x, "support_strength", 0.0), reverse=True)
    new_inactive.sort(key=lambda x: get_field(x, "support_strength", 0.0), reverse=True)
    
    # Helper for formatting list
    def format_arg_list(args: List[ArgumentRecord]) -> str:
        if not args:
            return "_None_"
        
        lines = []
        for arg in args:
            polarity = get_field(arg, "polarity", "unknown")
            pol = POLARITY_DISPLAY.get(polarity, polarity).upper()
            claim_text = get_field(arg, "claim", "")
            source_role = get_field(arg, "source_role", "")
            source_marker = f" ({source_role})" if source_role else ""
            strength = get_field(arg, "support_strength", 0.0)
            lines.append(f"- **[{pol}]** {claim_text} (Str: {strength:.2f}){source_marker}")
        return "\n".join(lines)

    md = f"""# Memory Snapshot: {agent_name}

**Topic:**_{topic}_

## Stance Evolution
*   **Initial:** {initial_stance:.4f}
*   **Final:** {final_stance:.4f}
*   **Drift:** {drift_symbol}{drift:.4f}

## 1. New Arguments (Acquired)
These are arguments the agent heard during the debate and accepted into its world view.

### Active (Influencing Stance)
{format_arg_list(new_active)}

### Inactive (Refuted/Blocked)
{format_arg_list(new_inactive)}

## 2. Seeded Arguments (Original Knowledge)
These are the arguments the agent started with.

### Still Active
{format_arg_list(seeded_active)}

### Dropped / Refuted
{format_arg_list(seeded_inactive)}
"""
    return md
