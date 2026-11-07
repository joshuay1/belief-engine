import random
import json
from typing import List
from .argument_record import ArgumentRecord

def retrieve_arguments(arguments: List[ArgumentRecord], strategy: str = "mixed", k: int = 3, **kwargs) -> List[ArgumentRecord]:
    """
    Retrieves k arguments based on the specified strategy.
    
    Strategies:
    - "relevant" (Default): Uses LLM to select arguments semantically relevant to the current query (e.g. opponent's last point).
    - "role_aware": Prioritizes the strongest arguments regardless of polarity.
    - "weighted": Selects pro/con quotas from the current stance score.
    - "proportional": Selects pro/con quotas from the active memory composition.
    - "recent": Retrieves the most recently added arguments.
    - "mixed": Randomly samples from both pro and con.
    """
    if not arguments:
        return []

    if strategy == "recent":
        # Sort by timestamp descending
        sorted_args = sorted(arguments, key=lambda x: x.timestamp, reverse=True)
        return sorted_args[:k]

    elif strategy == "role_aware":
        # Retrieve the strongest arguments regardless of polarity
        sorted_args = sorted(arguments, key=lambda x: x.support_strength, reverse=True)
        return sorted_args[:k]

    elif strategy == "relevant":
        # Semantic relevance using LLM (since we don't have embeddings yet)
        if not kwargs.get("query") or not kwargs.get("backend"):
            # Fallback to role_aware if no query/backend
            sorted_args = sorted(arguments, key=lambda x: x.support_strength, reverse=True)
            return sorted_args[:k]

        query = kwargs["query"]
        backend = kwargs["backend"]
        
        # Quick filter: If we have too many arguments, maybe limit the candidates first?
        # For now, assume < 50 arguments, so we can pass them all.
        
        candidates = [arg.to_dict() for arg in arguments if arg.credence_relevant]
        
        prompt = f"""
        You are a retrieval engine.
        Select the top {k} arguments from the list below that are MOST RELEVANT to the Query.
        
        Query: "{query}"
        
        Arguments:
        {json.dumps(candidates, indent=2)}
        
        Instructions:
        1. Identify arguments that directly support, refute, or relate to the Query.
        2. Output a JSON list of the IDs of the selected arguments.
        """
        
        try:
            response = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a retrieval engine. Output ONLY JSON.",
                temperature=0.0
            )
            cleaned = response.replace("```json", "").replace("```", "").strip()
            selected_ids = json.loads(cleaned)
            
            # Return the actual objects
            relevant_args = [arg for arg in arguments if arg.id in selected_ids]
            
            # Sort by strength to ensure we pick the most confident beliefs among the relevant ones
            relevant_args.sort(key=lambda x: x.support_strength, reverse=True)
            
            # If LLM returns fewer than k, fill with strongest remaining
            if len(relevant_args) < k:
                remaining = [arg for arg in arguments if arg not in relevant_args]
                remaining_sorted = sorted(remaining, key=lambda x: x.support_strength, reverse=True)
                relevant_args.extend(remaining_sorted[:k - len(relevant_args)])
                
            return relevant_args[:k]
            
        except Exception as e:
            print(f"   (Retrieval) Relevance check failed: {e}")
            # Fallback
            sorted_args = sorted(arguments, key=lambda x: x.support_strength, reverse=True)
            return sorted_args[:k]

    elif strategy == "proportional":
        from .argument_record import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE

        # The response context is defined over active evidence only. Archived
        # records must not influence the quota denominator or re-enter through
        # the backfill path.
        active_arguments = [
            arg for arg in arguments if getattr(arg, "credence_relevant", False)
        ]
        if not active_arguments:
            return []

        # Retrieve arguments proportional to the distribution in memory
        pro_args = [
            arg
            for arg in active_arguments
            if arg.polarity == POLARITY_AFFIRMATIVE
        ]
        con_args = [
            arg
            for arg in active_arguments
            if arg.polarity == POLARITY_NEGATIVE
        ]

        total_count = len(active_arguments)
            
        pro_ratio = len(pro_args) / total_count
        
        # Calculate target counts
        target_pro = int(round(k * pro_ratio))
        target_con = k - target_pro
        
        # Sort by strength
        pro_args.sort(key=lambda x: x.support_strength, reverse=True)
        con_args.sort(key=lambda x: x.support_strength, reverse=True)
        
        # Select
        selected = []
        selected.extend(pro_args[:target_pro])
        selected.extend(con_args[:target_con])
        
        # If we didn't get enough (e.g. ran out of one side), fill with strongest remaining
        if len(selected) < k:
            remaining = [arg for arg in active_arguments if arg not in selected]
            remaining.sort(key=lambda x: x.support_strength, reverse=True)
            selected.extend(remaining[:k - len(selected)])
            
        return selected[:k]

    elif strategy == "weighted":
        from .argument_record import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
        
        # Retrieve arguments conditioned on the current stance.
        current_stance = kwargs.get("current_stance", 0.0)
        
        # Map stance (-1 to 1) to Pro Probability (0 to 1)
        # Stance -1 -> 0.0 Pro
        # Stance 0 -> 0.5 Pro
        # Stance 1 -> 1.0 Pro
        pro_prob = (current_stance + 1.0) / 2.0
        
        # Calculate target counts
        target_pro = int(round(k * pro_prob))
        target_con = k - target_pro
        
        pro_args = [arg for arg in arguments if arg.polarity == POLARITY_AFFIRMATIVE and arg.credence_relevant]
        con_args = [arg for arg in arguments if arg.polarity == POLARITY_NEGATIVE and arg.credence_relevant]
        
        # Sort by strength
        pro_args.sort(key=lambda x: x.support_strength, reverse=True)
        con_args.sort(key=lambda x: x.support_strength, reverse=True)
        
        # Select
        selected = []
        selected.extend(pro_args[:target_pro])
        selected.extend(con_args[:target_con])
        
        # If we didn't get enough (e.g. ran out of one side), fill with strongest remaining
        if len(selected) < k:
            remaining = [arg for arg in arguments if arg not in selected]
            remaining.sort(key=lambda x: x.support_strength, reverse=True)
            selected.extend(remaining[:k - len(selected)])
            
        return selected[:k]

    else: # "mixed" or default
        # Simple random sample
        if len(arguments) <= k:
            return arguments
        return random.sample(arguments, k)
