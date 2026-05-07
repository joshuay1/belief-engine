import os
from datetime import datetime
from .argument_record import ArgumentRecord
from .memory_store import MemoryStore
import random

from typing import List, Dict, Optional, Any, Tuple, Union

# Wrapper class to expose MemoryStore API using PatchedAgenticMemorySystem
# AgenticMemorySystem taken from https://github.com/WujiangXu/A-mem-sys
from .amem.retrievers import CustomChromaRetriever
import json
import subprocess
from ..backend.backend_base import LLMBackend
import logging

logger = logging.getLogger(__name__)

os.makedirs("tempfiles", exist_ok=True)
CHECKPOINT_PATH = "tempfiles/checkpointtester.jsonl"

class PatchedAgenticMemorySystem:
    """Core memory system that manages memory notes and their evolution.
    
    This system provides:
    - Memory creation, retrieval, update, and deletion
    - Content analysis and metadata extraction
    - Memory evolution and relationship management
    - Hybrid search capabilities
    """
    def __init__(self,
                 backend: LLMBackend,
                 id: str = '0',
                 model_name: str = 'all-MiniLM-L6-v2',
                 evo_threshold: int = 100,
                 chroma_port: int = 8002,
                 enable_evolution: bool = False): # Disabled by default to use explicit EvolutionStrategies
        self.model_name = model_name

        # Create a fresh retriever instance pointing to agent's Chroma server
        self.retriever = CustomChromaRetriever(id, collection_name="memories", model_name=self.model_name, chroma_port=chroma_port,
                                               checkpoint_path=CHECKPOINT_PATH)
        self.memories: Dict[Union[str, int], ArgumentRecord] = self.retriever.memories

        # Initialize LLM controller
        # self.llm_controller = LLMController(llm_backend, llm_model, api_key, sglang_host, sglang_port)
        self.llm_controller = backend
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.enable_evolution = enable_evolution

        # Evolution system prompt
        self._evolution_system_prompt = '''
                                You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
                                Analyze the the new memory note according to keywords and context, also with their several nearest neighbors memory.
                                Make decisions about its evolution.

                                The new memory context:
                                {context}
                                content: {content}
                                keywords: {keywords}

                                The nearest neighbors memories (each line starts with memory_id):
                                {nearest_neighbors_memories}

                                Based on this information, determine:
                                1. Should this memory be evolved? Consider its relationships with other memories.
                                2. What specific actions should be taken (strengthen, update_neighbor)?
                                   2.1 If choose to strengthen the connection, which memory should it be connected to? Use the memory_id from the neighbors above. Can you give the updated tags of this memory?
                                   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
                                Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
                                Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
                                The number of neighbors is {neighbor_number}.
                                Return your decision in JSON format with the following structure:
                                {{
                                    "should_evolve": True or False,
                                    "actions": ["strengthen", "update_neighbor"],
                                    "suggested_connections": ["memory_id_1", "memory_id_2", ...],
                                    "tags_to_update": ["tag_1",..."tag_n"],
                                    "new_context_neighborhood": ["new context",...,"new context"],
                                    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
                                }}
                                '''

    def arg2metadata(self, argument: ArgumentRecord):
        metadata = {
            "id": argument.id,
            "content": argument.content,
            "keywords": argument.keywords,
            "links": argument.links,
            "retrieval_count": argument.retrieval_count,
            "timestamp": argument.timestamp,
            "last_accessed": argument.last_accessed,
            "context": argument.context,
            "evolution_history": argument.evolution_history,
            "category": argument.category,
            "tags": argument.tags,
            "polarity": argument.polarity,
            "source_role": argument.source_role,
            "support_strength": argument.support_strength,
            "evidence_snippets": argument.evidence_snippets,
            "category_tags": argument.category_tags,
            "credence_relevant": argument.credence_relevant
        }
        return metadata

    def analyze_content(self, content: str) -> Dict:            
        """Analyze content using LLM to extract semantic metadata.
        
        Uses a language model to understand the content and extract:
        - Keywords: Important terms and concepts
        - Context: Overall domain or theme
        - Tags: Classification categories
        
        Args:
            content (str): The text content to analyze
            
        Returns:
            Dict: Contains extracted metadata with keys:
                - keywords: List[str]
                - context: str
                - tags: List[str]
        """
        prompt = """Generate a structured analysis of the following content by:
            1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
            2. Extracting core themes and contextual elements
            3. Creating relevant categorical tags

            Format the response as a JSON object:
            {
                "keywords": [
                    // several specific, distinct keywords that capture key concepts and terminology
                    // Order from most to least important
                    // Don't include keywords that are the name of the speaker or time
                    // At least three keywords, but don't be too redundant.
                ],
                "context": 
                    // one sentence summarizing:
                    // - Main topic/domain
                    // - Key arguments/points
                    // - Intended audience/purpose
                ,
                "tags": [
                    // several broad categories/themes for classification
                    // Include domain, format, and type tags
                    // At least three tags, but don't be too redundant.
                ]
            }

            Content for analysis:
            """ + content
        try:
            response = self.llm_controller.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "keywords": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "context": {
                                    "type": "string",
                                },
                                "tags": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                }
                            }
                        }
                    }})
            return json.loads(response)
        except Exception as e:
            print(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}

    def add_argument(self, argument_record) -> str:
        """
        Add a new memory from an ArgumentRecord. Parallels add_note, but uses ArgumentRecord directly.
        """
        # Copy ArgumentRecord fields
        note = argument_record
        note.content = note.claim
        note.keywords = note.category_tags
        note.tags = note.category_tags


        # LLM Analysis Enhancement: Auto-generate attributes using LLM if they are empty or default values
        needs_analysis = (
            not getattr(note, 'keywords', []) or
            getattr(note, 'context', 'General') == 'General' or
            not getattr(note, 'tags', [])
        )
        if needs_analysis:
            analysis = self.analyze_content(note.claim)
            if not getattr(note, 'keywords', []):
                note.keywords = analysis.get("keywords", [])
            if getattr(note, 'context', 'General') == 'General':
                note.context = analysis.get("context", "General")
            if not getattr(note, 'tags', []):
                note.tags = analysis.get("tags", [])

        evo_label = False
        if self.enable_evolution and note.source_role!="seed":
            # Only run internal evolution if enabled
            evo_label, note = self.process_memory(note)
            
        self.memories[note.id] = note
        
        # Add to ChromaDB with complete metadata
        metadata = self.arg2metadata(note)

        self.retriever.add_document(note.claim, metadata, note.id)

        if evo_label == True:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id
    
    def consolidate_memories(self):
        """Consolidate memories: update retriever with new documents"""
        # Reset ChromaDB collection
        self.retriever.client.reset()
        
        # Re-add all memory documents with their complete metadata
        for memory in self.memories.values():
            metadata = self.arg2metadata(memory)
            self.retriever.add_document(memory.content, metadata, memory.id)
    
    def find_related_memories(self, query: str, k: int = 5) -> Tuple[str, List[str]]:
        """Find related memories using ChromaDB retrieval

        Returns:
            Tuple[str, List[str]]: (formatted_memory_string, list_of_memory_ids)
        """
        if not self.memories:
            return "", []

        try:
            # Get results from ChromaDB
            results = self.retriever.search(query, k)

            # Convert to list of memories
            memory_str = ""
            memory_ids = []

            if 'ids' in results and results['ids'] and len(results['ids'][0]) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Get metadata from ChromaDB results
                    if i < len(results['metadatas'][0]):
                        metadata = results['metadatas'][0][i]
                        # Format memory string with actual memory ID
                        memory_str += f"memory_id:{doc_id}\ttalk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                        memory_ids.append(doc_id)

            return memory_str, memory_ids
        except Exception as e:
            logger.error(f"Error in find_related_memories: {str(e)}")
            return "", []

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories using ChromaDB retrieval in raw format"""
        if not self.memories:
            return ""
            
        # Get results from ChromaDB
        results = self.retriever.search(query, k)
        
        # Convert to list of memories
        memory_str = ""
        
        if 'ids' in results and results['ids'] and len(results['ids']) > 0:
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if i < len(results['metadatas'][0]):
                    # Get metadata from ChromaDB results
                    metadata = results['metadatas'][0][i]
                    
                    # Add main memory info
                    memory_str += f"talk start time:{metadata.get('timestamp', '')}\tmemory content: {metadata.get('content', '')}\tmemory context: {metadata.get('context', '')}\tmemory keywords: {str(metadata.get('keywords', []))}\tmemory tags: {str(metadata.get('tags', []))}\n"
                    
                    # Add linked memories if available
                    links = metadata.get('links', [])
                    j = 0
                    for link_id in links:
                        if link_id in self.memories and j < k:
                            neighbor = self.memories[link_id]
                            memory_str += f"talk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
                            j += 1
                            
        return memory_str

    def read(self, memory_id: str) -> Optional[ArgumentRecord]:
        """Retrieve a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to retrieve
            
        Returns:
            ArgumentRecord if found, None otherwise
        """
        return self.memories.get(memory_id)
    
    def update(self, memory_id: str, **kwargs) -> bool:
        """Update a memory note.
        
        Args:
            memory_id: ID of memory to update
            **kwargs: Fields to update
            
        Returns:
            bool: True if update successful
        """
        if memory_id not in self.memories:
            return False
            
        note = self.memories[memory_id]
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
                
        # Update in ChromaDB
        metadata = self.arg2metadata(note)
        
        # Delete and re-add to update
        self.retriever.delete_document(memory_id)
        self.retriever.add_document(document=note.content, metadata=metadata, doc_id=memory_id)
        
        return True
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory note by its ID.
        
        Args:
            memory_id (str): ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted, False if not found
        """
        if memory_id in self.memories:
            # Delete from ChromaDB
            self.retriever.delete_document(memory_id)
            # Delete from local storage
            try:
                del self.memories[memory_id]
            except KeyError: # already deleted by a different process
                pass
            return True
        return False
    
    def _search_raw(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Internal search method that returns raw results from ChromaDB.
        
        This is used internally by the memory evolution system to find
        related memories for potential evolution.
        
        Args:
            query (str): The search query text
            k (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: Raw search results from ChromaDB
        """
        results = self.retriever.search(query, k)
        return [{'id': doc_id, 'score': score} 
                for doc_id, score in zip(results['ids'][0], results['distances'][0])]
                
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach."""
        # Get results from ChromaDB (only do this once)
        search_results = self.retriever.search(query, k)
        memories = []

        # Process ChromaDB results
        for i, doc_id in enumerate(search_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'tags': memory.tags,
                    'score': search_results['distances'][0][i]
                })

        return memories[:k]
    
    def _search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for memories using a hybrid retrieval approach.

        This method combines results from both:
        1. ChromaDB vector store (semantic similarity)
        2. Embedding-based retrieval (dense vectors)

        The results are deduplicated and ranked by relevance.

        Args:
            query (str): The search query text
            k (int): Maximum number of results to return

        Returns:
            List[Dict[str, Any]]: List of search results, each containing:
                - id: Memory ID
                - content: Memory content
                - context: Memory context
                - keywords: Memory keywords
                - tags: Memory tags
                - score: Similarity score
        """
        # Get results from ChromaDB
        chroma_results = self.retriever.search(query, k)
        memories = []

        # Process ChromaDB results
        for i, doc_id in enumerate(chroma_results['ids'][0]):
            memory = self.memories.get(doc_id)
            if memory:
                memories.append({
                    'id': doc_id,
                    'content': memory.content,
                    'context': memory.context,
                    'keywords': memory.keywords,
                    'tags': memory.tags,
                    'score': chroma_results['distances'][0][i]
                })

        # Get results from embedding retriever
        embedding_results = self.retriever.search(query, k)

        # Combine results with deduplication
        seen_ids = set(m['id'] for m in memories)
        for result in embedding_results:
            memory_id = result.get('id')
            if memory_id and memory_id not in seen_ids:
                memory = self.memories.get(memory_id)
                if memory:
                    memories.append({
                        'id': memory_id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'tags': memory.tags,
                        'score': result.get('score', 0.0)
                    })
                    seen_ids.add(memory_id)

        return memories[:k]

    def search_agentic(self, query: str, k: int = 5, where=None) -> List[Dict[str, Any]]:
        """Search for memories using ChromaDB retrieval."""
        if not self.memories:
            return []
            
        try:
            # Check availability to avoid annoying Chroma errors
            count = self.retriever.collection.count()
            effective_k = min(k, count) if count > 0 else 0
            
            if effective_k == 0:
                return []
                
            # Get results from ChromaDB
            results = self.retriever.search(query, effective_k, where=where)
            
            # Process results
            memories = []
            seen_ids = set()
            
            # Check if we have valid results
            if ('ids' not in results or not results['ids'] or 
                len(results['ids']) == 0 or len(results['ids'][0]) == 0):
                return []
                
            # Process ChromaDB results
            for i, doc_id in enumerate(results['ids'][0][:k]):
                if doc_id in seen_ids:
                    continue
                    
                if i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                    
                    # Create result dictionary with all metadata fields
                    memory_dict = {
                        'id': doc_id,
                        'content': metadata.get('content', ''),
                        'context': metadata.get('context', ''),
                        'keywords': metadata.get('keywords', []),
                        'tags': metadata.get('tags', []),
                        'timestamp': metadata.get('timestamp', ''),
                        'category': metadata.get('category', 'Uncategorized'),
                        'is_neighbor': False
                    }
                    
                    # Add score if available
                    if 'distances' in results and len(results['distances']) > 0 and i < len(results['distances'][0]):
                        memory_dict['score'] = results['distances'][0][i]
                        
                    memories.append(memory_dict)
                    seen_ids.add(doc_id)
            
            # Add linked memories (neighbors)
            neighbor_count = 0
            for memory in list(memories):  # Use a copy to avoid modification during iteration
                if neighbor_count >= k:
                    break
                    
                # Get links from metadata
                links = memory.get('links', [])
                if not links and 'id' in memory:
                    # Try to get links from memory object
                    mem_obj = self.memories.get(memory['id'])
                    if mem_obj:
                        links = mem_obj.links
                        
                for link_id in links:
                    if link_id not in seen_ids and neighbor_count < k:
                        neighbor = self.memories.get(link_id)
                        if neighbor:
                            memories.append({
                                'id': link_id,
                                'content': neighbor.content,
                                'context': neighbor.context,
                                'keywords': neighbor.keywords,
                                'tags': neighbor.tags,
                                'timestamp': neighbor.timestamp,
                                'category': neighbor.category,
                                'is_neighbor': True
                            })
                            seen_ids.add(link_id)
                            neighbor_count += 1
            
            return memories[:k]
        except Exception as e:
            logger.error(f"Error in search_agentic: {str(e)}")
            return []

    def process_memory(self, note: ArgumentRecord) -> Tuple[bool, ArgumentRecord]:
        """Process a memory note and determine if it should evolve.

        Args:
            note: The memory note to process

        Returns:
            Tuple[bool, ArgumentRecord]: (should_evolve, processed_note)
        """
        # For first memory or testing, just return the note without evolution
        if not self.memories:
            return False, note

        try:
            # Get nearest neighbors - now returns actual memory IDs
            neighbors_text, memory_ids = self.find_related_memories(note.content, k=5)
            if not neighbors_text or not memory_ids:
                return False, note

            # Format neighbors for LLM - in this case, neighbors_text is already formatted

            # Query LLM for evolution decision
            prompt = self._evolution_system_prompt.format(
                content=note.content,
                context=note.context,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbors_text,
                neighbor_number=len(memory_ids)
            )
            
            try:
                response = self.llm_controller.get_completion(
                    prompt,
                    response_format={"type": "json_schema", "json_schema": {
                        "name": "response",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "should_evolve": {
                                    "type": "boolean"
                                },
                                "actions": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "suggested_connections": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_context_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "tags_to_update": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    }
                                },
                                "new_tags_neighborhood": {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        }
                                    }
                                }
                            },
                            "required": ["should_evolve", "actions", "suggested_connections", 
                                      "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }}
                )
                
                response_json = json.loads(response)
                should_evolve = response_json["should_evolve"]
                
                if should_evolve:
                    actions = response_json["actions"]
                    for action in actions:
                        if action == "strengthen":
                            suggest_connections = response_json["suggested_connections"]
                            new_tags = response_json["tags_to_update"]
                            note.links.extend(suggest_connections)
                            note.tags = new_tags
                        elif action == "update_neighbor":
                            new_context_neighborhood = response_json["new_context_neighborhood"]
                            new_tags_neighborhood = response_json["new_tags_neighborhood"]

                            # Update each neighbor memory using its actual ID
                            for i in range(min(len(memory_ids), len(new_tags_neighborhood))):
                                memory_id = memory_ids[i]

                                # Skip if memory doesn't exist
                                if memory_id not in self.memories:
                                    continue

                                # Get the memory to update
                                neighbor_memory = self.memories[memory_id]

                                # Update tags
                                if i < len(new_tags_neighborhood):
                                    neighbor_memory.tags = new_tags_neighborhood[i]

                                # Update context
                                if i < len(new_context_neighborhood):
                                    neighbor_memory.context = new_context_neighborhood[i]

                                # Save the updated memory back
                                self.memories[memory_id] = neighbor_memory

                return should_evolve, note
                
            except (json.JSONDecodeError, KeyError, Exception) as e:
                logger.error(f"Error in memory evolution: {str(e)}")
                return False, note
                
        except Exception as e:
            # For testing purposes, catch all exceptions and return the original note
            logger.error(f"Error in process_memory: {str(e)}")
            return False, note
                 
class AgenticMemoryStore(MemoryStore):
    """
    Wraps PatchedAgenticMemorySystem to provide the MemoryStore API.
    """
    def __init__(self, backend: LLMBackend, llm_backend: str = "openai", llm_model: str = "qwen3:14b",  # noqa: kept for API compat
                 agent_id: str="default", **kwargs):
        super().__init__(**kwargs)
        self.agent_id = agent_id
        # self.topic_id = topic_id
        self.memory_system = PatchedAgenticMemorySystem(
            backend=backend,
            id=agent_id,
            model_name="all-MiniLM-L6-v2",
            enable_evolution=kwargs.get("enable_evolution", True) # Default to True to keep Amem features 
        )
        self.arguments = self.memory_system.memories

    def close(self):
        """Terminate the Chroma server for this agent."""
        self.memory_system.retriever.close()

    def seed_arguments(self, arguments: List[ArgumentRecord]):
        if self.memory_system.retriever.populated:
            return

        # Filter arguments by similarity threshold before seeding
        # Use the existing credence_manager from parent MemoryStore
        credence_manager = self.credence_manager

        # Get existing arguments in memory (should be empty at seeding, but check anyway)
        existing_args = list(self.arguments.values())

        # Filter arguments by similarity threshold
        # Only keep arguments that are sufficiently different from existing ones
        accepted_args = []

        # Filter arguments by similarity threshold before seeding
        for candidate in arguments:
            # Check similarity against existing memory
            similar_to_existing = self.credence_manager.find_similar_arguments(candidate, existing_args)
            
            # Check similarity against already accepted arguments in this batch
            similar_to_accepted = self.credence_manager.find_similar_arguments(candidate, accepted_args)
            
            if not similar_to_existing and not similar_to_accepted:
                # Sufficiently different - accept it
                accepted_args.append(candidate)
                existing_args.append(candidate)  # Add to existing for next comparisons
        
        # Add all accepted arguments to memory
        for argument in accepted_args:
            argument.source_role = "seed"
            # Set credence_relevant to True for all seeded arguments
            argument.credence_relevant = True

            self.memory_system.add_argument(argument)

            # Update belief engine for seeded arguments (critical for initial stance calculation)
            if self.stance_method in ("bayesian", "agentic", "llm") and argument.credence_relevant:
                delta = self.belief_engine.update(argument, backend=None)
                print(f"[SEEDING DEBUG] Updated belief engine with {argument.polarity} arg (strength={argument.support_strength:.2f}), delta={delta:.4f}, new_log_odds={self.belief_engine.log_odds:.4f}")

        msys = self.memory_system.retriever
        call = f'''cdp export "http://localhost:{msys.chroma_port}/{msys.collection_name}" > {CHECKPOINT_PATH}'''
        subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        self.memory_system.retriever.populated = True

    def add_argument(self, argument: ArgumentRecord, backend=None) -> (str, str, str, ArgumentRecord):
        pre_keys = set(self.arguments.keys())
        status, reason, snippet, argument = super().add_argument(argument, backend)
        post_keys = set(self.arguments.keys())

        # need to remove and re-add the new and replaced args
        keys_to_remove = pre_keys ^ post_keys

        assert status in ['added', 'replaced', 'ignored'], f"UNSUPPORTED STATUS: {status}"
        if status=='added': assert len(keys_to_remove)==1, print(keys_to_remove, argument.id)
        if status in ['added', 'replaced', 'synthesized']: # not "ignored"
            # for key in keys_to_remove:
            #remove from memory
            if keys_to_remove:
                self.memory_system.retriever.collection.delete(ids=list(keys_to_remove))

            # Add to agentic memory system (credence_relevant is already set by parent class)
            self.memory_system.add_argument(argument)
        
        return status, reason, snippet, argument

    def add_arguments_batch(self, arguments, backend=None):
        results = []
        belief_updated = False

        for argument in arguments:
            pre_keys = set(self.arguments.keys())
            status, reason, snippet, argument, updated = self._add_argument_internal(
                argument,
                backend=backend,
                persist_belief_state=False,
                trace_memory=False,
            )
            post_keys = set(self.arguments.keys())
            keys_to_remove = pre_keys ^ post_keys

            assert status in ['added', 'replaced', 'ignored'], f"UNSUPPORTED STATUS: {status}"
            if status == 'added':
                assert len(keys_to_remove) == 1, print(keys_to_remove, argument.id)

            if status in ['added', 'replaced', 'synthesized']:
                if keys_to_remove:
                    self.memory_system.retriever.collection.delete(ids=list(keys_to_remove))
                self.memory_system.add_argument(argument)

            results.append((status, reason, snippet, argument))
            belief_updated = belief_updated or updated

        if belief_updated and hasattr(self.belief_engine, "save_state"):
            self.belief_engine.save_state()
            self.log_memory_state()

        return results
    
    def retrieve(self, k=3, **kwargs) -> List[ArgumentRecord]:
        # If weighted, we need to pass the current stance
        if self.retrieval_strategy == "weighted":
            kwargs["current_stance"] = self.get_stance()

        strategy = self.retrieval_strategy
        arguments = list(self.arguments.values())

        if strategy == "recent":
            # Sort by timestamp descending
            sorted_args = sorted(arguments, key=lambda x: x.timestamp, reverse=True)
            return sorted_args[:k]

        elif strategy == "role_aware":
            # Retrieve the strongest arguments regardless of polarity
            sorted_args = sorted(arguments, key=lambda x: x.support_strength, reverse=True)
            return sorted_args[:k]

        elif strategy=='relevant':
            # Use agentic search
            return [self.memory_system.memories[m['id']] for m in self.memory_system.search_agentic(kwargs.get('query', ""), k=k)]

        elif strategy=='proportional':
            from .argument_record import POLARITY_AFFIRMATIVE, POLARITY_NEGATIVE
        
            # Retrieve arguments proportional to the distribution in memory
            pro_args_ids = self.memory_system.search_agentic(kwargs.get('query', ""), k=k, where={'$and': [{'polarity': POLARITY_AFFIRMATIVE},
                                                                                                  {'credence_relevant': True}]})
            con_args_ids = self.memory_system.search_agentic(kwargs.get('query', ""), k=k, where={'$and': [{'polarity': POLARITY_NEGATIVE},
                                                                                                  {'credence_relevant': True}]})
            pro_args = [self.arguments[m['id']] for m in pro_args_ids]
            con_args = [self.arguments[m['id']] for m in con_args_ids]

            
            total_count = len(self.arguments)
            if total_count == 0:
                return []
                
            total_pro_count = sum(1 for arg in self.arguments.values() if arg.polarity == POLARITY_AFFIRMATIVE and arg.credence_relevant)
            total_con_count = sum(1 for arg in self.arguments.values() if arg.polarity == POLARITY_NEGATIVE and arg.credence_relevant)
            total_relevant = total_pro_count + total_con_count
            
            if total_relevant == 0:
                pro_ratio = 0.5 # Default to equal if empty
            else:
                pro_ratio = total_pro_count / total_relevant
            
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
                remaining = [arg for arg in self.arguments.values() if arg not in selected]
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
            
            # Use agentic search to get pro and con arguments
            pro_args_ids = self.memory_system.search_agentic(kwargs.get('query', ""), k=k, where={'$and': [{'polarity': POLARITY_AFFIRMATIVE},
                                                                                                  {'credence_relevant': True}]})
            con_args_ids = self.memory_system.search_agentic(kwargs.get('query', ""), k=k, where={'$and': [{'polarity': POLARITY_NEGATIVE},
                                                                                                  {'credence_relevant': True}]})
            pro_args = [self.arguments[m['id']] for m in pro_args_ids]
            con_args = [self.arguments[m['id']] for m in con_args_ids]
            
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

        else:  
            if len(arguments) <= k:
                return arguments
            return random.sample(arguments, k)
