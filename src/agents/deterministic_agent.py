from typing import List, Dict, Any, Tuple
import random
from src.simulation.agent_interface import DebatingAgent
from src.utils.json_data_loader import load_arguments_from_json
from src.constants import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE

class DeterministicAgent(DebatingAgent):
    """
    An agent that deterministically (based on a random seed) returns human-authored 
    arguments from a dataset, without using an LLM to generate text.
    Acts as a 'Human Baseline'.
    """
    def __init__(self, 
                 name: str, 
                 topic: str, 
                 side: str, 
                 seed: int = 42, 
                 limit: int = 50,
                 data_config: Dict[str, Any] = None):
        """
        Args:
            name: Display name.
            topic: Debate topic string.
            side: 'affirmative' or 'negative'.
            seed: Random seed for argument selection order.
            limit: Number of arguments to load and shuffle.
            data_config: Config for data loading (path, etc.).
        """
        self._name = name
        self.topic = topic
        self.side = side
        self.seed = seed
        self.limit = limit
        
        # Load arguments
        self.arguments = self._load_arguments(data_config)
        
        # Shuffle deterministically
        self.rng = random.Random(seed)
        self.rng.shuffle(self.arguments)
        
        self.current_index = 0
        
    @property
    def name(self) -> str:
        return self._name

    def _load_arguments(self, data_config: Dict[str, Any]) -> List[str]:
        # Determine path
        json_path = "data/Argument-Quality-Ranking-30k/arguments.json"
        if data_config and "json_path" in data_config:
            json_path = data_config["json_path"]
            
        # Determine polarity
        polarity = POLARITY_AFFIRMATIVE if self.side in [POLARITY_AFFIRMATIVE, "pro"] else POLARITY_NEGATIVE
        
        try:
            # Using existing loader
            # Strategy 'random' or 'quality_max'? Or just load all?
            # User wants "random seed in there", implies randomness.
            # We load a batch, then shuffle them ourselves.
            # Loader returns ArgumentRecord-like dicts or objects?
            # `load_arguments_from_json` returns list of dicts or records?
            # Let's import it and check its return type.
            # Assuming it returns list of dicts based on usages in orchestrator.
            
            # We use 'random' strategy to get a diverse set initially, then shuffle?
            # Or 'quality_max' to get best?
            # Default to 'quality_max' unless specified.
            strategy = "quality_max"
            
            args = load_arguments_from_json(
                json_path=json_path,
                topic=self.topic,
                side=polarity,
                limit=self.limit,
                strategy=strategy
            )
            
            # Extract just the text (claim + stance/reasoning?)
            # Usually 'claim' is the argument text.
            return [arg.claim for arg in args]
            
        except Exception as e:
            print(f"[WARN] DeterministicAgent failed to load arguments: {e}")
            return [f"I have no arguments prepared for {self.topic}."]

    def generate_reply(self, conversation_history: List[Dict[str, str]]) -> Tuple[str, Any]:
        """
        Returns the next argument in the shuffled list.
        Loops back to start if exhausted.
        """
        if not self.arguments:
            return "I have nothing further to add.", {}
            
        arg_text = self.arguments[self.current_index]
        
        # Increment index
        self.current_index = (self.current_index + 1) % len(self.arguments)
        
        return arg_text, {"source": "deterministic_dataset", "index": self.current_index}

    def get_current_stance(self) -> float:
        """
        Returns fixed stance based on side.
        """
        return 1.0 if self.side in [POLARITY_AFFIRMATIVE, "pro"] else -1.0

    def get_stance(self) -> float:
        return self.get_current_stance()
