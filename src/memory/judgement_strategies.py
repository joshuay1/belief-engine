from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from .argument_record import ArgumentRecord
from .credence_manager import CredenceManager
from ..utils.output import print_debug
import json

class JudgementStrategy(ABC):
    """
    Abstract base class for memory judgement strategies.
    Determines how new arguments interact with existing memory (filter, replace, merge).
    """
    
    @abstractmethod
    def process(self, new_argument: ArgumentRecord, existing_arguments: Dict[str, ArgumentRecord], backend=None) -> Tuple[str, Optional[str], str, ArgumentRecord]:
        """
        Process a new argument against existing memory.
        
        Args:
            new_argument: The argument to ingest.
            existing_arguments: The current state of memory (id -> record).
            backend: Optional LLM backend for logic-based strategies.
            
        Returns:
            (status, target_id, reason, record)
            - status: "added", "replaced", "ignored"
            - target_id: ID of the existing argument that was "replaced" (hidden), or None
            - reason: Explanation for the action
            - record: The argument record to be stored
        """
        pass

class VectorJudgementStrategy(JudgementStrategy):
    """
    Math-based strategy using Vector Similarity.
    Philosophy: "If it looks the same mathematically, hide the weaker one."
    
    Args:
        similarity_threshold: Threshold for considering arguments similar (0.0-1.0)
        self_similarity_threshold: Stricter threshold for self-generated arguments
        cross_polarity_dedup: If True, compare against ALL arguments regardless of polarity.
                              If False (default), only compare against same-polarity arguments.
    """
    def __init__(self, similarity_threshold: float = 0.85, self_similarity_threshold: float = 0.95, cross_polarity_dedup: bool = False):
        self.similarity_threshold = similarity_threshold
        self.self_similarity_threshold = self_similarity_threshold
        self.cross_polarity_dedup = cross_polarity_dedup
        # Note: CredenceManager is used for the vector search logic
        self.credence_manager = CredenceManager(similarity_threshold=similarity_threshold)
        
    def process(self, new_argument: ArgumentRecord, existing_arguments: Dict[str, ArgumentRecord], backend=None) -> Tuple[str, Optional[str], str, ArgumentRecord]:
        # 1. Filter: Determine which arguments to compare against
        if self.cross_polarity_dedup:
            # NEW: Compare against ALL active arguments (semantic deduplication across polarities)
            # This recognizes that "Social media harms mental health" and "Social media improves mental health"
            # are about the SAME dimension and should interact.
            relevant_args = [
                arg for arg in existing_arguments.values() 
                if arg.credence_relevant
            ]
        else:
            # ORIGINAL: Only compare against same-polarity arguments
            # Treats opposing arguments as fundamentally different things.
            relevant_args = [
                arg for arg in existing_arguments.values() 
                if arg.credence_relevant and arg.polarity == new_argument.polarity
            ]
        
        # Determine strictness based on source
        threshold_to_use = self.similarity_threshold
        source_role = getattr(new_argument, 'source_role', 'opponent')
        
        if source_role == 'self':
            threshold_to_use = self.self_similarity_threshold
            
        # 2. Check for similarity
        similar_args = self.credence_manager.find_similar_arguments(new_argument, relevant_args, threshold=threshold_to_use)
        
        if not similar_args:
            # Novel argument
            new_argument.credence_relevant = True
            return "added", None, "Novel argument (Vector distinct)", new_argument
        
        # 3. Compare with the SINGLE most similar neighbor
        most_similar_arg, similarity_score = similar_args[0]
        
        # Check if this is a cross-polarity match (opposing argument on same dimension)
        is_cross_polarity = (most_similar_arg.polarity != new_argument.polarity)
        
        if is_cross_polarity and self.cross_polarity_dedup:
            # Cross-polarity semantic match: These are COUNTER-ARGUMENTS on the same dimension
            # Strategy: The newer argument doesn't replace, but we acknowledge the dimension is "covered"
            # We IGNORE the new argument because we already have reasoning on this dimension
            new_argument.credence_relevant = False
            reason = f"Vector Cross-Polarity Ignore: Same dimension already covered by [{most_similar_arg.polarity}] arg [Sim: {similarity_score:.2f}]"
            return "ignored", None, reason, new_argument
        
        # Same-polarity match: Standard strength comparison
        if new_argument.support_strength > most_similar_arg.support_strength:
            # New is stronger -> Soft Replace Old
            most_similar_arg.credence_relevant = False
            new_argument.credence_relevant = True
            
            reason = f"Vector Replace: New ({new_argument.support_strength:.2f}) > Old ({most_similar_arg.support_strength:.2f}) [Sim: {similarity_score:.2f}]"
            return "replaced", most_similar_arg.id, reason, new_argument
        else:
            # New is weaker -> Soft Ignore New
            new_argument.credence_relevant = False
            
            reason = f"Vector Ignore: New ({new_argument.support_strength:.2f}) <= Old ({most_similar_arg.support_strength:.2f}) [Sim: {similarity_score:.2f}]"
            return "ignored", None, reason, new_argument

class AgenticJudgementStrategy(JudgementStrategy):
    """
    Logic-based strategy using LLM reasoning (Reasoning/Agentic).
    Philosophy: "If it means the same thing logically, hide the weaker one."
    """
    def __init__(self):
        # Initialize CredenceManager for RAG-based pre-filtering
        # We use a low threshold (0.5) to cast a wide net, ensuring we catch potential duplicates
        # even if they aren't perfect vector matches.
        self.credence_manager = CredenceManager(similarity_threshold=0.5)
        
    def process(self, new_argument: ArgumentRecord, existing_arguments: Dict[str, ArgumentRecord], backend=None) -> Tuple[str, Optional[str], str, ArgumentRecord]:
        if not existing_arguments:
            new_argument.credence_relevant = True
            return "added", None, "First argument", new_argument
            
        if not backend:
            new_argument.credence_relevant = True
            return "added", None, "No backend for LLM strategy", new_argument

        # 1. Retrieval (RAG): Select top-k candidates to reduce context window
        # We search across ALL arguments (Active + Refuted) to ensure we detect duplicates of hidden args.
        all_args_list = list(existing_arguments.values())
        
        # Threshold=0.0 means "return everything sorted by similarity"
        ranked_candidates = self.credence_manager.find_similar_arguments(
            new_argument, 
            all_args_list, 
            threshold=0.0
        )
        
        # Take Top-15 most similar arguments for the Context Window
        # (15 args * ~50 tokens = 750 tokens context, very safe)
        top_k = 15
        retrieved_args = [item[0] for item in ranked_candidates[:top_k]]
        
        existing_claims = []
        for arg in retrieved_args:
            status_label = "ACTIVE" if arg.credence_relevant else "REFUTED/INACTIVE"
            existing_claims.append({
                "id": arg.id,
                "claim": arg.claim,
                "strength": arg.support_strength,
                "status": status_label
            })
            
        print_debug(f"AgenticJudgement (RAG): {len(all_args_list)} args -> {len(existing_claims)} candidates for LLM")

        if not existing_claims:
            new_argument.credence_relevant = True
            return "added", None, "First argument", new_argument

        # 2. Identify Semantic Equivalence & Quality Comparison
        prompt = f"""
        You are an Intellectual Exploration Engine.
        Your goal is to build a rich, multi-faceted understanding of a topic by including nuanced perspectives.
        
        New Argument: "{new_argument.claim}" (Strength: {new_argument.support_strength})
        
        Existing Beliefs (Context):
        {json.dumps(existing_claims, indent=2)}
        
        Instructions:
        1. Check if the New Argument is a DUPLICATE of any existing belief (ACTIVE or REFUTED).
        2. If it is a DUPLICATE of a REFUTED/INACTIVE belief:
           - ACTION: IGNORE
           - REASON: "Duplicate of previously refuted argument [ID]"
        3. If it is a DUPLICATE of an ACTIVE belief:
           - If New is better articulated -> ACTION: REPLACE [Existing ID]
           - If Old is much better -> ACTION: IGNORE
        4. If it is NOVEL and adds high-quality nuance -> ACTION: ADD
           
        Output format:
        ACTION: [ADD / REPLACE / IGNORE]
        TARGET_ID: [ID of existing argument, or None]
        REASON: [Brief explanation]
        """
        
        print_debug(f"AgenticJudgement comparing against {len(existing_claims)} args")
        
        try:
            response = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a strict knowledge filter. Output ACTION, TARGET_ID, REASON.",
                temperature=0.0
            ).strip()
            
            lines = response.split('\n')
            action = "added"
            target_id = None
            reason = "LLM Decision"
            
            for line in lines:
                if line.startswith("ACTION:"):
                    raw_action = line.replace("ACTION:", "").strip().upper()
                    if "REPLACE" in raw_action: action = "replaced"
                    elif "IGNORE" in raw_action: action = "ignored"
                    else: action = "added"
                if line.startswith("TARGET_ID:"):
                    tid = line.replace("TARGET_ID:", "").strip()
                    if tid and tid.lower() != "none":
                        target_id = tid
                if line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            
            # Execute Soft Logic
            if action == "replaced" and target_id:
                # Find the old argument object
                old_arg = existing_arguments.get(target_id)
                if old_arg:
                    old_arg.credence_relevant = False
                    new_argument.credence_relevant = True
                    return "replaced", target_id, reason, new_argument
                else:
                    # Target ID invalid? Fallback to add
                    new_argument.credence_relevant = True
                    return "added", None, f"LLM returned invalid target ID {target_id}", new_argument
            
            elif action == "ignored":
                new_argument.credence_relevant = False
                return "ignored", None, reason, new_argument
                
            else:
                # Add
                new_argument.credence_relevant = True
                return "added", None, reason, new_argument

        except Exception as e:
            print_debug(f"Error in LLM Evolution: {e}")
            new_argument.credence_relevant = True
            return "added", None, f"Error: {e}", new_argument
