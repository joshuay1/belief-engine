"""
Credence Relevance Manager

Manages the credence_relevant tag for arguments based on semantic similarity
and argument quality scores. This module handles embedding arguments and
determining which arguments should be marked as credence_relevant.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .argument_record import ArgumentRecord


class CredenceManager:
    """
    Manages credence relevance tagging for arguments.
    
    Uses sentence transformers to embed arguments and calculate cosine similarity.
    Determines credence relevance based on similarity thresholds and quality scores.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Credence Manager.
        
        Args:
            similarity_threshold: Threshold for considering arguments similar (default: 0.7)
            model_name: Name of the sentence transformer model (default: "all-MiniLM-L6-v2")
        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self._model = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
    
    def _get_model(self) -> SentenceTransformer:
        """Lazy load the sentence transformer model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for a text string, using cache if available.
        
        Args:
            text: The text to embed
            
        Returns:
            numpy array of the embedding vector
        """
        # Use text as cache key
        if text not in self._embedding_cache:
            model = self._get_model()
            embedding = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
            self._embedding_cache[text] = embedding
        return self._embedding_cache[text]
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar_arguments(
        self, 
        new_argument: ArgumentRecord,
        existing_arguments: List[ArgumentRecord],
        threshold: Optional[float] = None
    ) -> List[Tuple[ArgumentRecord, float]]:
        """
        Find existing arguments that are similar to the new argument.
        
        Args:
            new_argument: The new argument to compare
            existing_arguments: List of existing arguments to compare against
            threshold: Optional threshold override. If None, uses self.similarity_threshold.
            
        Returns:
            List of tuples (argument, similarity_score) for arguments above threshold,
            sorted by similarity (highest first)
        """
        if not existing_arguments:
            return []
        
        # Use provided threshold or default
        effective_threshold = threshold if threshold is not None else self.similarity_threshold
        
        # Get embedding for new argument
        new_embedding = self._get_embedding(new_argument.claim)
        
        # Calculate similarities
        similar_args = []
        for existing_arg in existing_arguments:
            existing_embedding = self._get_embedding(existing_arg.claim)
            similarity = self.cosine_similarity(new_embedding, existing_embedding)
            
            if similarity >= effective_threshold:
                similar_args.append((existing_arg, similarity))
        
        # Sort by similarity (highest first)
        similar_args.sort(key=lambda x: x[1], reverse=True)
        return similar_args
    
    def update_credence_relevance(
        self,
        new_argument: ArgumentRecord,
        existing_arguments: Dict[str, ArgumentRecord]
    ) -> None:
        """
        Update credence_relevant tags for new and existing arguments.
        
        Logic:
        1. Find all credence_relevant=True arguments in existing memory
        2. Calculate similarity to new argument
        3. If no similar arguments (above threshold): tag new as credence_relevant=True
        4. If similar arguments found: do pairwise comparison with each similar argument
           - For each similar argument:
             * If new > similar: set similar.credence_relevant = False
             * If new < similar: keep similar.credence_relevant = True (unchanged)
             * If new == similar: both can be credence_relevant
           - If new is stronger than at least one similar argument: tag new as credence_relevant=True
           - If new is weaker than all similar arguments: tag new as credence_relevant=False
        
        Args:
            new_argument: The new argument being added
            existing_arguments: Dictionary of existing arguments (id -> ArgumentRecord)
        """
        # Print extracted argument details
        claim_preview = new_argument.claim[:60] + "..." if len(new_argument.claim) > 60 else new_argument.claim
        print(f"\n   [Credence Check]")
        print(f"   New argument: \"{claim_preview}\"")
        print(f"   Strength: {new_argument.support_strength:.4f}")
        print(f"   Checking for similar arguments...")
        
        # Get all credence_relevant arguments from existing memory
        credence_relevant_args = [
            arg for arg in existing_arguments.values()
            if getattr(arg, 'credence_relevant', False)
        ]
        
        # Find similar arguments
        similar_args = self.find_similar_arguments(new_argument, credence_relevant_args)
        
        if not similar_args:
            # No similar arguments found above threshold
            # Tag new argument as credence_relevant
            new_argument.credence_relevant = True
            print(f"   Result: No similar arguments found → credence_relevant = True")
        else:
            # Found similar argument(s) - do pairwise comparison with each
            new_quality = new_argument.support_strength
            new_is_stronger_than_any = False
            
            print(f"   Found {len(similar_args)} similar argument(s):")
            
            # Compare new argument pairwise with each similar argument
            for i, (similar_arg, similarity_score) in enumerate(similar_args, 1):
                similar_quality = similar_arg.support_strength
                similar_claim_preview = similar_arg.claim[:60] + "..." if len(similar_arg.claim) > 60 else similar_arg.claim
                
                print(f"      [{i}] \"{similar_claim_preview}\"")
                print(f"          Strength: {similar_quality:.4f} | Similarity: {similarity_score:.3f}")
                
                if new_quality > similar_quality:
                    # New argument is stronger than this similar argument
                    # Set this similar argument to False
                    similar_arg.credence_relevant = False
                    new_is_stronger_than_any = True
                    print(f"          Comparison: New ({new_quality:.4f}) > Old ({similar_quality:.4f}) → Replacing the old argument")
                elif new_quality < similar_quality:
                    # This similar argument is stronger than new
                    # Keep this similar argument as True (it was already True)
                    print(f"          Comparison: New ({new_quality:.4f}) < Old ({similar_quality:.4f}) → Keeping the old argument")
                else:
                    # Equal quality - both can be credence_relevant
                    # Keep similar as True, and mark that new should be True too
                    new_is_stronger_than_any = True
                    print(f"          Comparison: New ({new_quality:.4f}) == Old ({similar_quality:.4f}) → Both can are credence_relevant")
            
            # Set new argument's credence_relevant based on whether it was stronger than any
            new_argument.credence_relevant = new_is_stronger_than_any
            result_str = "True" if new_is_stronger_than_any else "False"
            reason = "stronger than at least one similar argument" if new_is_stronger_than_any else "weaker than all similar arguments"
            print(f"   Result: credence_relevant = {result_str} ({reason})")
    
