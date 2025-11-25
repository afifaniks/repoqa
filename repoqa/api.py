# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

from typing import Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from repoqa.app import RepoQA
from repoqa.config import config
from repoqa.llm.llm_factory import get_llm
from repoqa.storage.collection_manager import (
    collection_exists_and_has_documents,
    delete_collection,
    get_collection_name,
)
from repoqa.util.setup_util import setup

setup()

app = FastAPI(
    title=config.api_title,
    description=config.api_description,
    version=config.api_version,
)


class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    repo: str = Field(
        ...,
        description="Repository URL or path to analyze",
        example="https://github.com/afifaniks/repoqa.git",
    )
    question: str = Field(
        ..., description="Question to ask about the repository", min_length=1
    )
    mode: str = Field(
        default=config.pipeline_mode,
        description="Mode of operation: 'agent' or 'rag'",
    )
    llm_model: Optional[str] = Field(
        default=config.llm_model, description="LLM model to use"
    )
    force_update: Optional[bool] = Field(
        default=False,
        description="Force re-indexing by deleting and recreating collection",
    )


class AnswerResponse(BaseModel):
    """Response model for answers."""

    question: str
    answer: str
    repo: str


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "OK",
        "service": config.api_title,
        "version": config.api_version,
    }


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about a repository.

    Args:
        request: Question request with repo, question, and options

    Returns:
        Answer response with the generated answer
    """
    try:
        # Generate collection name for this repository
        collection_name = get_collection_name(request.repo)
        logger.info(f"Using collection '{collection_name}' for repo: {request.repo}")

        # Handle force_update: delete existing collection if it exists
        if request.force_update:
            logger.info(
                f"Force update requested, deleting collection " f"'{collection_name}'"
            )
            delete_collection(config.vectorstore_persist_directory, collection_name)

        # Check if collection already exists with documents
        collection_has_data = collection_exists_and_has_documents(
            config.vectorstore_persist_directory, collection_name
        )

        # Create new RepoQA instance for each request
        logger.info(f"Initializing RepoQA for repo: {request.repo}")
        llm_model = request.llm_model or config.llm_model
        repo_qa_instance = RepoQA(
            persist_directory=config.vectorstore_persist_directory,
            embedding_model=config.embedding_model,
            collection_name=collection_name,
            collection_chunk_size=config.vectorstore_chunk_size,
            llm_model=get_llm(
                llm_model, backend=config.llm_backend, kwargs={"mode": request.mode}
            ),
            mode=request.mode,
            repo_path=config.repository_clone_directory,
            ollama_base_url=config.ollama_base_url,
            temperature=config.llm_temperature,
        )

        # Index repository if collection doesn't exist or force_update is True
        if request.force_update or not collection_has_data:
            logger.info(f"Indexing repository: {request.repo}")
            result = repo_qa_instance.index_repository(
                repo_path=request.repo,
                clone_dir=config.repository_clone_directory,
            )
            logger.info(f"Indexing completed: {result}")
        else:
            logger.info(
                f"Collection '{collection_name}' already exists with data, "
                "skipping indexing"
            )

        # Ask question
        logger.info(f"Processing question: {request.question}")
        answer = repo_qa_instance.ask(request.question)

        return AnswerResponse(
            question=request.question,
            answer=answer,
            repo=request.repo,
        )

    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
    }
