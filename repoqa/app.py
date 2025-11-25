# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from repoqa.embedding.sentence_transformer import SentenceTransformerEmbedding
from repoqa.indexing.git_indexer import GitRepoIndexer
from repoqa.llm.llm_factory import get_llm
from repoqa.util.setup_util import setup

setup()

from repoqa.pipeline.agentic_rag import AgenticRAGPipeline  # noqa: E402
from repoqa.pipeline.rag import RAGPipeline  # noqa: E402


class RepoQA:
    """Main RepoQA application class for code-based question answering."""

    def __init__(
        self,
        llm_model: Any,
        embedding_model: str,
        collection_name: str,
        collection_chunk_size: int,
        ollama_base_url: str,
        mode: str,
        repo_path: str,
        persist_directory: str,
        temperature: float = 0.3,
    ):
        """Initialize RepoQA with customizable components.

        Args:
            llm_model: Name of the LLM model to use.
            embedding_model: Name of the embedding model.
            collection_name: Name of the vector store collection.
            collection_chunk_size: Chunk size for document splitting.
            ollama_base_url: Base URL for Ollama server.
            mode: Operation mode, either 'agent' or 'rag'.
            repo_path: Path to the repository for agentic operations.
            persist_directory: Directory to persist vector store data.
            temperature: Sampling temperature for LLM responses.
        """
        self.mode = mode

        repo_indexer = GitRepoIndexer(
            SentenceTransformerEmbedding(model_name=embedding_model),
            chunk_size=collection_chunk_size,
        )

        if mode == "agent":
            logger.info("Initializing Agent pipeline...")
            self.pipeline = AgenticRAGPipeline(
                llm_model=llm_model,
                embedding_model=embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name,
                ollama_base_url=ollama_base_url,
                temperature=temperature,
                repo_path=repo_path,
                repo_indexer=repo_indexer,
            )
        elif mode == "rag":
            logger.info("Initializing RAG pipeline...")
            self.pipeline = RAGPipeline(
                llm_model=llm_model,
                embedding_model=embedding_model,
                persist_directory=persist_directory,
                collection_name=collection_name,
                ollama_base_url=ollama_base_url,
                temperature=temperature,
                repo_indexer=repo_indexer,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def index_repository(
        self,
        repo_path: Union[str, Path],
        clone_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index a repository and store embeddings.

        Args:
            repo_path: Path/URL of the repository to index.
            clone_dir: Optional directory to clone into.

        Returns:
            Dictionary with indexing results and metadata.
        """
        return self.pipeline.index_repository(repo_path, clone_dir)

    def ask(self, query: str) -> str:
        """Answer a question about the repository.

        Args:
            query: Natural language query about the repository.

        Returns:
            Generated answer based on repository context.
        """
        return self.pipeline.ask(query)
