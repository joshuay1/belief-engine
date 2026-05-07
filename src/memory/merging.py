from typing import Dict, Tuple
from .argument_record import ArgumentRecord

def merge_arguments(existing_args: Dict[str, ArgumentRecord], new_arg: ArgumentRecord, similarity_threshold: float = 0.8) -> Tuple[Dict[str, ArgumentRecord], str]:
    """
    Merges a new argument into the existing dict if a similar one exists.
    Otherwise, adds it to the dict.
    
    Returns:
        Tuple[Dict[str, ArgumentRecord], str]: (Updated dict, Status string 'merged' or 'added' or 'ignored')
    """
    
    # Check for duplicates
    for arg in existing_args.values():
        # Simple merging logic: if claims are identical or very similar
        if arg.claim.lower() == new_arg.claim.lower() and arg.polarity == new_arg.polarity:
            # Disable reinforcement: Do not update strength or evidence
            # Just mark as ignored (duplicate)
            return existing_args, "ignored"
        
    # If not found, add it
    existing_args[new_arg.id] = new_arg

    return existing_args, "added"
