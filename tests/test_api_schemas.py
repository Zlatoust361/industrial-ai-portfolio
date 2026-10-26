"""Unit tests for `src.api.schemas` validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import EmbedResponse, TextEmbedRequest


def test_text_embed_request_requires_at_least_one_text() -> None:
    with pytest.raises(ValidationError):
        TextEmbedRequest(texts=[])

    request = TextEmbedRequest(texts=["a defective weld"])
    assert request.texts == ["a defective weld"]


def test_embed_response_round_trip() -> None:
    response = EmbedResponse(vectors=[[0.1, 0.2]], model_name="fake-model", dim=2)
    assert response.dim == 2
    assert response.model_dump()["model_name"] == "fake-model"
