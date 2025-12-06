import grpc
import pytest
import math
import numbers

from tests.embedding_pb2 import (
    EmbedTextRequest,
    EmbedBatchRequest,
    Empty,
)
from tests.embedding_pb2_grpc import EmbeddingServiceStub


@pytest.fixture(scope="module")
def channel() -> grpc.Channel:
    return grpc.insecure_channel("localhost:8351")


@pytest.fixture(scope="module")
def stub(channel: grpc.Channel) -> EmbeddingServiceStub:
    return EmbeddingServiceStub(channel)


@pytest.fixture(scope="module")
def embedding_dimension(stub: EmbeddingServiceStub) -> int:
    response = stub.GetEmbeddingDimension(Empty())
    assert isinstance(response.dimension, int)
    assert response.dimension > 0
    return response.dimension


def _is_unit_norm(vector: list[float], rel_tol: float = 1e-3) -> bool:
    norm = math.sqrt(sum(v * v for v in vector))
    return math.isclose(norm, 1.0, rel_tol=rel_tol)


def test_health(stub: EmbeddingServiceStub) -> None:
    response = stub.HealthCheck(Empty())
    assert response.status == "SERVING"
    assert isinstance(response.model_name, str)
    assert isinstance(response.device, str)


def test_get_embedding_dimension(stub: EmbeddingServiceStub) -> None:
    response = stub.GetEmbeddingDimension(Empty())
    assert isinstance(response.dimension, int)
    assert response.dimension > 0


def test_embed_text(stub: EmbeddingServiceStub, embedding_dimension: int) -> None:
    request = EmbedTextRequest(
        text="Hello world",
        max_length=512,
        normalize=True,
        pooling_strategy="mean",
    )
    response = stub.EmbedText(request)
    embedding = list(response.embedding)
    assert isinstance(embedding, list)
    assert len(embedding) == embedding_dimension
    assert all(isinstance(x, numbers.Real) for x in embedding)
    assert _is_unit_norm(embedding)


def test_embed_batch(stub: EmbeddingServiceStub, embedding_dimension: int) -> None:
    request = EmbedBatchRequest(
        texts=["Hello world", "This is a test", "Machine learning is awesome"],
        batch_size=8,
        max_length=512,
        normalize=True,
        pooling_strategy="mean",
    )
    response = stub.EmbedBatch(request)
    assert len(response.embeddings) == 3
    for emb in response.embeddings:
        vector = list(emb.vector)
        assert isinstance(vector, list)
        assert len(vector) == embedding_dimension
        assert all(isinstance(x, numbers.Real) for x in vector)
        assert _is_unit_norm(vector)


@pytest.mark.parametrize("strategy", ["mean", "cls", "max"])
def test_pooling_strategies(stub: EmbeddingServiceStub, embedding_dimension: int, strategy: str) -> None:
    request = EmbedTextRequest(
        text="This is a test sentence for pooling strategies.",
        max_length=512,
        normalize=True,
        pooling_strategy=strategy,
    )
    response = stub.EmbedText(request)
    embedding = list(response.embedding)
    assert isinstance(embedding, list)
    assert len(embedding) == embedding_dimension
    assert all(isinstance(x, numbers.Real) for x in embedding)
    assert _is_unit_norm(embedding)


def test_error_empty_text(stub: EmbeddingServiceStub) -> None:
    request = EmbedTextRequest(
        text="",
        max_length=512,
        normalize=True,
        pooling_strategy="mean",
    )
    with pytest.raises(Exception):
        _ = stub.EmbedText(request)


def test_error_invalid_pooling(stub: EmbeddingServiceStub) -> None:
    request = EmbedTextRequest(
        text="test",
        max_length=512,
        normalize=True,
        pooling_strategy="invalid",
    )
    with pytest.raises(Exception):
        _ = stub.EmbedText(request)
