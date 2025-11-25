# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Afif Al Mamun

"""Unit tests for LLM factory."""

from unittest.mock import Mock, patch

import pytest
import requests


class TestLLMFactory:
    """Test suite for LLM factory."""

    @patch("repoqa.llm.llm_factory.requests.get")
    @patch("repoqa.llm.llm_factory.OllamaLLM")
    def test_get_llm_existing_model(self, mock_ollama, mock_get):
        """Test getting LLM with existing model."""
        from repoqa.llm.llm_factory import get_llm

        # Mock response for existing models
        mock_response = Mock()
        mock_response.json.return_value = {
            "models": [
                {"name": "qwen3:1.7b"},
                {"name": "llama3.2:3b"},
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        mock_llm_instance = Mock()
        mock_ollama.return_value = mock_llm_instance

        llm = get_llm("qwen3:1.7b", backend="ollama")

        assert llm == mock_llm_instance
        mock_ollama.assert_called_once_with(
            model="qwen3:1.7b",
            base_url="http://localhost:11434",
            temperature=0.5,
            reasoning=False,
            num_ctx=16000,
        )

    @patch("repoqa.llm.llm_factory.requests.post")
    @patch("repoqa.llm.llm_factory.requests.get")
    @patch("repoqa.llm.llm_factory.OllamaLLM")
    def test_get_llm_pull_missing_model(self, mock_ollama, mock_get, mock_post):
        """Test getting LLM and pulling missing model."""
        from repoqa.llm.llm_factory import get_llm

        # Mock response for no existing models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"models": []}
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock post response for pulling model
        mock_post_response = Mock()
        mock_post_response.status_code = 200
        mock_post_response.iter_lines.return_value = [
            '{"status": "downloading"}',
            '{"status": "complete"}',
        ]
        mock_post.return_value = mock_post_response

        mock_llm_instance = Mock()
        mock_ollama.return_value = mock_llm_instance

        llm = get_llm("new-model:latest", backend="ollama")

        assert llm == mock_llm_instance
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/pull",
            json={"name": "new-model:latest"},
            stream=True,
        )

    @patch("repoqa.llm.llm_factory.requests.get")
    def test_get_llm_connection_error(self, mock_get):
        """Test handling of connection errors."""
        from repoqa.llm.llm_factory import get_llm

        mock_get.side_effect = requests.exceptions.ConnectionError("Connection refused")

        with pytest.raises(ConnectionError) as exc_info:
            get_llm("qwen3:1.7b", backend="ollama")

        assert "Failed to connect to Ollama server" in str(exc_info.value)

    @patch("repoqa.llm.llm_factory.requests.post")
    @patch("repoqa.llm.llm_factory.requests.get")
    def test_get_llm_pull_failure(self, mock_get, mock_post):
        """Test handling of model pull failures."""
        from repoqa.llm.llm_factory import get_llm

        # Mock response for no existing models
        mock_get_response = Mock()
        mock_get_response.json.return_value = {"models": []}
        mock_get_response.raise_for_status = Mock()
        mock_get.return_value = mock_get_response

        # Mock failed pull response
        mock_post_response = Mock()
        mock_post_response.status_code = 404
        mock_post_response.text = "Model not found"
        mock_post.return_value = mock_post_response

        with pytest.raises(ValueError) as exc_info:
            get_llm("nonexistent:model", backend="ollama")

        assert "Failed to pull model" in str(exc_info.value)

    def test_get_llm_unsupported_backend(self):
        """Test handling of unsupported backend."""
        from repoqa.llm.llm_factory import get_llm

        with pytest.raises(ValueError) as exc_info:
            get_llm("test-model", backend="unsupported")

        assert "Unsupported backend" in str(exc_info.value)

    @patch("repoqa.llm.llm_factory.requests.get")
    @patch("repoqa.llm.llm_factory.OllamaLLM")
    def test_get_llm_custom_kwargs(self, mock_ollama, mock_get):
        """Test getting LLM with custom kwargs."""
        from repoqa.llm.llm_factory import get_llm

        # Mock response for existing models
        mock_response = Mock()
        mock_response.json.return_value = {"models": [{"name": "custom-model"}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        mock_llm_instance = Mock()
        mock_ollama.return_value = mock_llm_instance

        llm = get_llm(
            "custom-model",
            backend="ollama",
            kwargs={
                "base_url": "http://custom:8080",
                "temperature": 0.8,
            },
        )

        assert llm == mock_llm_instance
        mock_ollama.assert_called_once_with(
            model="custom-model",
            base_url="http://custom:8080",
            temperature=0.8,
            reasoning=False,
            num_ctx=16000,
        )
