import numbers

import grpc
import pytest

from tests.reranker_pb2 import Empty, RerankRequest
from tests.reranker_pb2_grpc import RerankerServiceStub


@pytest.fixture(scope="module")
def channel() -> grpc.Channel:
    return grpc.insecure_channel("localhost:8352")


@pytest.fixture(scope="module")
def stub(channel: grpc.Channel) -> RerankerServiceStub:
    return RerankerServiceStub(channel)


def test_health(stub: RerankerServiceStub) -> None:
    response = stub.HealthCheck(Empty())
    assert response.status == "SERVING"
    assert isinstance(response.model_name, str)
    assert isinstance(response.device, str)


def test_rerank_orders_scores_by_relevance(stub: RerankerServiceStub) -> None:
    request = RerankRequest(
        query="What is the capital of France?",
        texts=[
            "Paris is the capital of France.",
            "Bananas are a good source of potassium.",
            "The Eiffel Tower is located in Paris.",
        ],
    )
    response = stub.Rerank(request)
    scores = list(response.scores)
    assert len(scores) == 3
    assert all(isinstance(score, numbers.Real) for score in scores)
    assert scores[0] > scores[1]
    assert scores[2] > scores[1]


def test_error_empty_query(stub: RerankerServiceStub) -> None:
    request = RerankRequest(query="", texts=["some text"])
    with pytest.raises(Exception):
        _ = stub.Rerank(request)


def test_error_empty_texts(stub: RerankerServiceStub) -> None:
    request = RerankRequest(query="some query", texts=[])
    with pytest.raises(Exception):
        _ = stub.Rerank(request)
