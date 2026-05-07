import os
import datetime
import uuid
from typing import List, Dict, Any, Optional
import json
import logging
import chromadb
from chromadb.utils import embedding_functions
from ..simulation.belief_engine import create_belief_engine
from .argument_record import ArgumentRecord

# Silence noisy Chroma telemetry logger paths globally for this process.
for _logger_name in (
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry.posthog",
    "chromadb.telemetry",
    "posthog",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

class MoltbotMemory:
    """
    File-based memory system with ChromaDB indexing.
    Replicates the Moltbot architecture:
    - Daily Markdown Logs (Short-term)
    - MEMORY.md (Long-term)
    - Hybrid/Semantic Retrieval via Chroma
    """
    def __init__(self, agent_name: str, root_dir: str = "agent_workspace", allow_self_arguments: bool = True, belief_engine_type: str = "bayesian", stance_method: str = "bayesian", backend=None, storage_mode="episodic", worldview_openness="balanced"):
        self.agent_name = agent_name
        self.root_dir = root_dir
        self.backend = backend
        self.belief_engine_type = belief_engine_type
        self.stance_method = stance_method
        self.storage_mode = storage_mode # Added
        self.worldview_openness = worldview_openness # Added
        
        self.allow_self_arguments = allow_self_arguments
        
        # Paths
        self.memory_dir = os.path.join(root_dir, "memory", agent_name)
        self.daily_dir = os.path.join(self.memory_dir, "daily")
        self.chroma_dir = os.path.join(self.memory_dir, "chroma_db")
        self.memory_md_path = os.path.join(self.memory_dir, "MEMORY.md")
        self.soul_md_path = os.path.join(self.memory_dir, "SOUL.md")
        
        # Ensure directories exist
        os.makedirs(self.daily_dir, exist_ok=True)
        
        # Initialize Belief Engine (Standardized)
        self.belief_engine = create_belief_engine(
            agent_name=agent_name,
            memory_dir=self.memory_dir,
            config={"belief_engine_type": self.belief_engine_type}
        )
        
        # Linear Mass Tracking (for 'blind' / 'linear' mode)
        self.pro_mass = 0.0
        self.con_mass = 0.0
        
        # Initialize ChromaDB (Local Persistent)
        try:
            from chromadb.config import Settings
            # Try new API (0.4+)
            try:
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
            except TypeError:
                # Older signatures may not accept `settings` in PersistentClient
                self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        except AttributeError:
            # Fallback for older versions (0.3.x)
            from chromadb.config import Settings
            self.chroma_client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.chroma_dir,
                anonymized_telemetry=False
            ))
        
        # Use default embedding function (all-MiniLM-L6-v2) implies we handle it automatically
        # Or explicit:
        try:
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            print(f"Warning: Could not load SentenceTransformer, falling back to default. Error: {e}")
            self.ef = None # Chroma default

        self.collection = self.chroma_client.get_or_create_collection(
            name=f"memory_{agent_name}",
            embedding_function=self.ef
        )


    # --- REFUTATION HELPER (Ported from MemoryStore) ---
    def _attempt_refutation(self, argument: Any, backend) -> (bool, str):
        """
        Checks if the agent's existing memory contains information that effectively refutes the new argument.
        Returns (True, Reason) if refuted.
        Returns (False, "") if not refuted.
        """
        # 0. Hard Check for Stubbornness (No LLM needed)
        if self.worldview_openness == "stubborn":
            # Calculate dominant stance using our stance method
            current_score = self.get_stance()
            current_stance = "affirmative" if current_score > 0 else "negative"
            arg_polarity = getattr(argument, 'polarity', 'unknown')

            # If new argument opposes current stance, reject it immediately
            if arg_polarity != "unknown" and arg_polarity != current_stance:
                return True, f"Dogmatic rejection: Argument ({arg_polarity}) opposes current worldview ({current_stance})."

        # Retrieve potentially relevant memories (RAG)
        # Use simple Vector Search (strategy="relevant")
        claim = getattr(argument, 'claim', str(argument))
        k_refutation = 15
        
        # Use our own retrieve method
        relevant_records = self.retrieve(query=claim, k=k_refutation, strategy="relevant")
        existing_claims = [r.claim for r in relevant_records]
        
        if not existing_claims:
            return False, ""
            
        print(f"   [Moltbot] Refutation Check (RAG): Checking against {len(existing_claims)} most relevant beliefs.")
        
        # Adjust strictness based on openness
        if self.worldview_openness == "open_minded":
            # High Openness: Hard to refute (requires direct proof)
            strictness_instruction = """
            STRICTNESS: LOW (Open-Minded).
            - Only refute if the 'Existing Knowledge' DIRECTLY and FACTUALLY disproves the 'New Argument'.
            - Allow nuances, different perspectives, and subjective disagreements.
            - If there is any ambiguity, DO NOT refute.
            """
        elif self.worldview_openness == "stubborn":
            # Low Openness: Easy to refute (stubborn)
            strictness_instruction = """
            STRICTNESS: EXTREME (Dogmatic).
            - You are a closed-minded defender of the 'Existing Knowledge'.
            - If the 'New Argument' conflicts with the 'Existing Knowledge' in ANY way (fact, sentiment, or conclusion), you MUST refute it.
            - If the 'New Argument' comes from an opposing stance, AUTOMATICALLY refute it using the 'Existing Knowledge'.
            - Do not allow any nuance. If it's not what we already believe, it is wrong.
            """
        else:
            # Medium Openness (Default: "balanced")
            strictness_instruction = """
            STRICTNESS: MEDIUM (Balanced).
            - Refute if the 'Existing Knowledge' provides a strong, logical counter-argument.
            - Do not refute if the 'New Argument' is a valid point that can coexist with existing knowledge.
            """

        prompt = f"""
        You are a critical reasoning engine.
        Determine if the "New Argument" is effectively REFUTED by the "Existing Knowledge".
        
        New Argument: "{claim}"
        
        Existing Knowledge:
        {json.dumps(existing_claims, indent=2)}
        
        {strictness_instruction}
        
        Instructions:
        1. Analyze if 'Existing Knowledge' invalidates 'New Argument' based on the STRICTNESS level above.
        2. If REFUTED: Output 'YES' followed by a brief explanation.
        3. If NOT REFUTED: Output 'NO'.
        
        Output format:
        YES: [Brief explanation]
        or
        NO
        """
        try:
            response = backend.chat(
                messages=[{"role": "user", "content": prompt}],
                system_prompt="You are a critical thinker. Output YES or NO.",
                temperature=0.0
            ).strip()
            
            if response.upper().startswith("YES"):
                reason = response[4:].strip()
                return True, reason
        except Exception as e:
            print(f"   (Memory) Refutation check failed: {e}")
        
        return False, ""

    def log_interaction(self, source: str, content: str, input_query: str = None):
        """
        Appends an interaction to today's daily log file.
        """
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        daily_file = os.path.join(self.daily_dir, f"{today}.md")
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        entry = f"\n## {timestamp} - {source}\n"
        if input_query:
            entry += f"> **Context:** {input_query}\n\n"
        entry += f"{content}\n"
        
        with open(daily_file, "a") as f:
            f.write(entry)
            
        # Immediate indexing for simplicity in this V1
        # In production this would be batched or watched
        self._index_text(content, source_file=f"daily/{today}.md", tags={"source": source, "type": "daily_log"})

    def promote_to_long_term(self, content: str, category: str = "General"):
        """
        Appends important information to MEMORY.md
        """
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Simple append structure
        entry = f"\n- **{timestamp} [{category}]:** {content}"
        
        # Ensure file starts with header if new
        if not os.path.exists(self.memory_md_path):
            with open(self.memory_md_path, "w") as f:
                f.write("# Long-term Memory\n")
        
        with open(self.memory_md_path, "a") as f:
            f.write(entry)
            
        self._index_text(content, source_file="MEMORY.md", tags={"source": "system", "type": "long_term", "category": category})

    def _index_text(self, text: str, source_file: str, tags: Dict[str, Any]):
        """
        Chunks and indexes text into ChromaDB.
        """
        # Simple paragraph chunking for now
        chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
        
        if not chunks:
            return

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = []
        for c in chunks:
            meta = {"source_file": source_file}
            meta.update(tags)
            metadatas.append(meta)
            
        self.collection.add(
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )

    def retrieve(self, query: str, k: int = 5, strategy: str = "weighted") -> List[ArgumentRecord]:
        """
        Semantic retrieval from memory.
        Returns ArgumentRecord objects conforming to the MemorySystem protocol.
        """
        if self.collection.count() == 0:
            return []
            
        # Helper to get all records for manual strategy filtering (since Chroma is only relevant)
        # For proportional/random, we need the full set or a large sample
        if strategy != "relevant":
             # Fetch all (or large limit)
             all_results = self.collection.get()
             if not all_results['ids']:
                 return []
             
             # Convert to ArgumentRecords
             all_records = []
             for i, id_val in enumerate(all_results['ids']):
                 doc = all_results['documents'][i]
                 meta = all_results['metadatas'][i] if all_results['metadatas'] else {}
                 record = ArgumentRecord(
                    id=id_val,
                    topic_id="memory",
                    claim=doc,
                    source_role=f"memory:{meta.get('source_file', 'unknown')}",
                    polarity=meta.get('polarity', 'neutral'), # Moltbot needs to store polarity in metadata for this to work!
                    support_strength=1.0,
                    credence_relevant=True,
                    timestamp=datetime.datetime.now()
                 )
                 all_records.append(record)
             
             # Use the standard retrieval logic from retrieval.py
             from .retrieval import retrieve_arguments
             return retrieve_arguments(
                 all_records,
                 strategy=strategy,
                 k=k,
                 query=query,
                 backend=self.backend,
                 current_stance=self.get_stance(),
             )

        # Default Chroma Logic (Relevant)
        count = self.collection.count()
        k = min(k, count)
        
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Flatten results (Chroma returns list of list)
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        ids = results['ids'][0] if results['ids'] else []
        
        records = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            source_file = meta.get('source_file', 'unknown')
            tags = meta # All metadata as tags
            
            # Create a generic argument record wrapping the text chunk
            # We map 'source_file' to 'source_role' loosely, or keep it in metadata
            record = ArgumentRecord(
                id=ids[i],
                topic_id="memory", # Default topic
                claim=doc,
                # We use the source file as the 'source' of the memory
                source_role=f"memory:{source_file}", 
                polarity="neutral",
                support_strength=1.0,
                credence_relevant=False,
                timestamp=datetime.datetime.now() # ArgumentRecord expects datetime object, not string
            )
            # Attach extras if ArgumentRecord allows (it's a dataclass, so maybe not dynamic atts without slot issues)
            # But standard fields are enough.
            records.append(record)
            
        return records

    def add_argument(self, argument: Any, backend=None):
        """
        Adds an argument to memory and updates the Belief Engine.
        """
        if not self.allow_self_arguments and getattr(argument, "source_role", "") == "self":
            return "ignored", "Self arguments disabled", str(argument)[:60], argument

        # argument is likely an ArgumentRecord object or dict
        source_role = getattr(argument, 'source_role', 'unknown')
        
        # WORLDVIEW FILTRATION CHECK (Replaces old Reframing)
        # Skip refutation check for SEED arguments (initialization)
        if self.storage_mode == "worldview" and source_role != "seed":
             engine_backend = backend or self.backend
             if engine_backend:
                 print(f"!!! MOLTBOT WORLDVIEW CHECK !!!")
                 is_refuted, reason = self._attempt_refutation(argument, engine_backend)
                 if is_refuted:
                     print(f"!!! MOLTBOT REJECTED ARGUMENT !!!: {reason}")
                     return "ignored", f"Refuted by Worldview: {reason}", str(argument)[:60], argument


        claim = getattr(argument, 'claim', str(argument))
        pol = getattr(argument, 'polarity', 'unknown')
        strength = getattr(argument, 'support_strength', 1.0)
        
        # Update Linear Mass
        if pol == "affirmative":
            self.pro_mass += strength
        elif pol == "negative":
            self.con_mass += strength
        
        if self.stance_method in ("bayesian", "agentic", "llm"):
            # 1. Update Belief Engine (The Core Math)
            # We need to make sure 'argument' has the fields BeliefEngine needs (strength, polarity)
            if hasattr(argument, 'support_strength'):
                engine_backend = backend or self.backend
                
                # Retrieve context relevant to the new argument to see if we already know about this
                # or if we have counter-arguments.
                context_records = self.retrieve(query=claim, k=3)
                context_str = "\n".join([f"- {r.claim} (Role: {r.source_role})" for r in context_records])
                
                delta = self.belief_engine.update(argument, backend=engine_backend, memory_context=context_str)
    
                # 2. Log the Belief Shift (Narrative)
                new_desc = self.belief_engine.get_belief_description()
                self.log_interaction("System (Belief)", f"Accepted argument: '{claim[:50]}...'\nStance Shift: {delta:+.3f}\nNew State: {new_desc}")
            else:
                self.log_interaction("System", f"noted: {claim}")
        else:
            # Blind/Linear Mode
            self.log_interaction("System (Blind)", f"Accepted argument: '{claim[:50]}...'\n(Belief Engine Bypassed)")
        
        # Immediate Indexing (for retrieval)
        # We index it as a "memory" type so it shows up in retrieval
        self._index_text(claim, source_file="short_term_memory", tags={"source": "add_argument", "type": "memory", "polarity": pol, "strength": strength})

        # Return compatible tuple: (status, reason, snippet, record)
        return "added", "Integrated into Belief Engine", claim[:60], argument

    def get_stance(self) -> float:
        """
        Returns the stance based on the configured method.
        """
        if self.stance_method in ("bayesian", "agentic", "llm"):
            return self.belief_engine.get_belief_score()
        else:
            # Linear / Blind
            total = self.pro_mass + self.con_mass
            if total > 0:
                return (self.pro_mass - self.con_mass) / total
            return 0.0

    def get_all_arguments(self) -> List[Dict[str, Any]]:
        """
        Returns memory contents in a format compatible with the orchestrator's evaluation.
        """
        all_text = ""
        
        # Include Long-Term Memory
        if os.path.exists(self.memory_md_path):
            with open(self.memory_md_path, "r") as f:
                all_text += f.read() + "\n\n"
                
        # Include Daily Logs (Last 3 days)
        # We don't want everything or it assumes too much context
        all_text += self.get_recent_context(days=3)
        
        # Return as a single 'argument' record to satisfy the list expectation
        # The evaluator effectively just reads the claim text anyway.
        return [{
            "id": "moltbot_memory_dump",
            "claim": all_text,
            "polarity": "neutral",
            "support_strength": 1.0, 
            "credence_relevant": True
        }]

    def get_recent_context(self, days: int = 1) -> str:
        """
        Reads the raw content of the last N daily logs.
        """
        today = datetime.date.today()
        context = ""
        
        for i in range(days + 1): # Today and past N days
            date = today - datetime.timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            filepath = os.path.join(self.daily_dir, f"{date_str}.md")
            
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    context += f"\n\n# Date: {date_str}\n{f.read()}"
        
        return context

    def initialize_soul(self, persona_text: str):
        """Writes the SOUL.md file if it doesn't exist."""
        if not os.path.exists(self.soul_md_path):
            with open(self.soul_md_path, "w") as f:
                f.write(persona_text)

    def read_soul(self) -> str:
        """Reads the SOUL.md file."""
        if os.path.exists(self.soul_md_path):
            with open(self.soul_md_path, "r") as f:
                return f.read()
        return ""

    @property
    def arguments(self) -> Dict[str, Any]:
        """
        Compat property: Returns a dict representation of memory.
        """
        # Return the dump as a single entry
        args_list = self.get_all_arguments()
        return {arg['id']: arg for arg in args_list}

    @property
    def credence_manager(self):
        """
        Dummy CredenceManager to satisfy Orchestrator seeding logic.
        Disables deduplication by always returning 'not similar'.
        """
        class DummyCredenceManager:
            def find_similar_arguments(self, *args, **kwargs):
                return False
            def cosine_similarity(self, *args, **kwargs):
                return 0.0
            def _get_embedding(self, text):
                return [0.0]*384
        return DummyCredenceManager()
