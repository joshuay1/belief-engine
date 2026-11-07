# AgenticMemorySystem taken from https://github.com/WujiangXu/A-mem-sys
import atexit
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import chromadb
import logging

logger = logging.getLogger(__name__)
for _logger_name in (
    "chromadb.telemetry.product.posthog",
    "chromadb.telemetry.posthog",
    "chromadb.telemetry",
    "posthog",
):
    logging.getLogger(_logger_name).setLevel(logging.CRITICAL)
from chromadb.utils.embedding_functions import \
    SentenceTransformerEmbeddingFunction

from ..argument_record import ArgumentRecord


class SilentSentenceTransformerEmbeddingFunction(SentenceTransformerEmbeddingFunction):
    def __call__(self, input):
        embeddings = self._model.encode(
            input, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()


class CustomChromaRetriever:  # lets us have concurrent agents running
    def __init__(self, id: str, collection_name: str = "memories", model_name: str = "all-MiniLM-L6-v2", chroma_port: int = 8002,
                 checkpoint_path='tempfiles/checkpoint_unified_json.jsonl'):
        self.populated = False
        self.chroma_process = None # Added for local client mode
        self.checkpoint_path = checkpoint_path
        self.chroma_port = chroma_port
        self.collection_name = collection_name

        # Use Local Client for Chroma 0.3.x compatibility
        self.chroma_path = f"tempfiles/chroma_db_{id}"
        
        # Fresh start: Clean up existing DB for this agent
        if os.path.exists(self.chroma_path):
            try:
                shutil.rmtree(self.chroma_path)
            except Exception as e:
                logger.warning(f"Warning: Failed to clean up {self.chroma_path}: {e}")

        logger.info(f"Initializing Local Chroma Client at {self.chroma_path}")
        from chromadb.config import Settings
        
        # New ChromaDB Client Initialization
        self.client = chromadb.PersistentClient(
            path=self.chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.embedding_function = SilentSentenceTransformerEmbeddingFunction(
            model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embedding_function)
        self.memories = {}
        atexit.register(self.close)

    def _start_chroma_server(self):
        """Start a Chroma server instance for this agent."""
        self.chroma_path = f"tempfiles/chroma_temp"
        if os.path.exists(self.chroma_path):
            shutil.rmtree(self.chroma_path)
        cmd = ['chroma', 'run', '--path', self.chroma_path,
               '--port', str(self.chroma_port)]
        logger.info(f"Starting Chroma server: {' '.join(cmd)}")
        self.chroma_process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(3)  # Give server time to start
        logger.info(f"Chroma server started for agent on port {self.chroma_port}")

    def close(self):
        if self.chroma_process:
            self.chroma_process.terminate()
            try:
                self.chroma_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.chroma_process.kill()
                self.chroma_process.wait()
            self.chroma_process = False

        # Force client release
        self.client = None
        import gc
        gc.collect()

        # Cleanup paths
        if os.path.exists(self.chroma_path):
            try:
                shutil.rmtree(self.chroma_path)
            except Exception as e:
                logger.warning(f"Warning: Failed to cleanup {self.chroma_path}: {e}")
                
        if os.path.exists(self.checkpoint_path):
            try:
                shutil.rmtree(self.checkpoint_path)
            except:
                pass

    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB with enhanced embedding using metadata.

        Args:
            document: Text content to add
            metadata: Dictionary of metadata including keywords, tags, context
            doc_id: Unique identifier for the document
        """
        # Build enhanced document content including semantic metadata
        enhanced_document = document

        # Add context information
        if 'context' in metadata and metadata['context'] != "General":
            enhanced_document += f" context: {metadata['context']}"

        # Add keywords information
        if 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords'] if isinstance(
                metadata['keywords'], list) else json.loads(metadata['keywords'])
            if keywords:
                enhanced_document += f" keywords: {', '.join(keywords)}"

        # Add tags information
        if 'tags' in metadata and metadata['tags']:
            tags = metadata['tags'] if isinstance(
                metadata['tags'], list) else json.loads(metadata['tags'])
            if tags:
                enhanced_document += f" tags: {', '.join(tags)}"

        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)

        # Store enhanced document content for better embedding
        processed_metadata['enhanced_content'] = enhanced_document

        # Use enhanced document content for embedding generation
        self.collection.add(
            documents=[enhanced_document],
            metadatas=[processed_metadata],
            ids=[doc_id]
        )

    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.

        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])

    def search(self, query: str, k: int = 5, where=None):
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where
        )

        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass

        return results
