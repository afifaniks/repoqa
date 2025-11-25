# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Integration tests for API endpoints."""

import sys
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

# Get reference to the global chromadb mock from sys.modules
chromadb_mock = sys.modules["chromadb"]


@pytest.fixture
def client():
    """Create a test client for the API."""
    # Import after mocking to avoid side effects
    with patch("repoqa.api.setup"):
        from repoqa.api import app

        return TestClient(app)


class TestAPI:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns status."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "OK"
        assert "service" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    @patch("repoqa.api.RepoQA")
    @patch("repoqa.api.collection_exists_and_has_documents")
    @patch("repoqa.api.get_llm")
    def test_ask_endpoint_new_repo(
        self, mock_get_llm, mock_collection_exists, mock_repoqa, client
    ):
        """Test ask endpoint with new repository."""
        # Mock collection doesn't exist
        mock_collection_exists.return_value = False

        # Mock RepoQA instance
        mock_instance = Mock()
        mock_instance.index_repository.return_value = {
            "status": "success",
            "documents_added": 10,
        }
        mock_instance.ask.return_value = "This is the answer."
        mock_repoqa.return_value = mock_instance

        # Mock LLM
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/ask",
            json={
                "repo": "https://github.com/test/repo.git",
                "question": "What is this repo about?",
                "mode": "rag",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "What is this repo about?"
        assert data["answer"] == "This is the answer."
        assert data["repo"] == "https://github.com/test/repo.git"

        # Verify indexing was called
        mock_instance.index_repository.assert_called_once()
        mock_instance.ask.assert_called_once_with("What is this repo about?")

    @patch("repoqa.api.RepoQA")
    @patch("repoqa.api.collection_exists_and_has_documents")
    @patch("repoqa.api.get_llm")
    def test_ask_endpoint_existing_repo(
        self, mock_get_llm, mock_collection_exists, mock_repoqa, client
    ):
        """Test ask endpoint with existing indexed repository."""
        # Mock collection exists
        mock_collection_exists.return_value = True

        # Mock RepoQA instance
        mock_instance = Mock()
        mock_instance.ask.return_value = "This is the answer."
        mock_repoqa.return_value = mock_instance

        # Mock LLM
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/ask",
            json={
                "repo": "https://github.com/test/repo.git",
                "question": "What is this repo about?",
                "mode": "rag",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify indexing was NOT called
        mock_instance.index_repository.assert_not_called()
        mock_instance.ask.assert_called_once()

    @patch("repoqa.api.RepoQA")
    @patch("repoqa.api.delete_collection")
    @patch("repoqa.api.collection_exists_and_has_documents")
    @patch("repoqa.api.get_llm")
    def test_ask_endpoint_force_update(
        self,
        mock_get_llm,
        mock_collection_exists,
        mock_delete_collection,
        mock_repoqa,
        client,
    ):
        """Test ask endpoint with force update."""
        # Mock collection exists but will be deleted
        mock_collection_exists.return_value = True
        mock_delete_collection.return_value = True

        # Mock RepoQA instance
        mock_instance = Mock()
        mock_instance.index_repository.return_value = {"status": "success"}
        mock_instance.ask.return_value = "Answer"
        mock_repoqa.return_value = mock_instance

        # Mock LLM
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/ask",
            json={
                "repo": "https://github.com/test/repo.git",
                "question": "Test question",
                "force_update": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Verify collection was deleted and re-indexed
        mock_delete_collection.assert_called_once()
        mock_instance.index_repository.assert_called_once()

    @patch("repoqa.api.RepoQA")
    @patch("repoqa.api.collection_exists_and_has_documents")
    @patch("repoqa.api.get_llm")
    def test_ask_endpoint_error_handling(
        self, mock_get_llm, mock_collection_exists, mock_repoqa, client
    ):
        """Test ask endpoint error handling."""
        mock_collection_exists.return_value = False

        # Mock RepoQA to raise an exception
        mock_repoqa.side_effect = Exception("Test error")

        # Mock LLM
        mock_llm = Mock()
        mock_get_llm.return_value = mock_llm

        response = client.post(
            "/ask",
            json={
                "repo": "https://github.com/test/repo.git",
                "question": "Test question",
            },
        )

        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]

    def test_ask_endpoint_validation(self, client):
        """Test ask endpoint input validation."""
        # Missing required fields
        response = client.post("/ask", json={})
        assert response.status_code == 422

        # Empty question
        response = client.post(
            "/ask",
            json={
                "repo": "https://github.com/test/repo.git",
                "question": "",
            },
        )
        assert response.status_code == 422
