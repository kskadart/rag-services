import argparse
import json
import logging

import grpc

from tests.reranker_pb2 import Empty, RerankRequest
from tests.reranker_pb2_grpc import RerankerServiceStub

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stub(host: str, port: int) -> RerankerServiceStub:
    channel = grpc.insecure_channel(f"{host}:{port}")
    return RerankerServiceStub(channel)


def cmd_health(stub: RerankerServiceStub) -> None:
    resp = stub.HealthCheck(Empty())
    data = {"status": resp.status, "model_name": resp.model_name, "device": resp.device}
    print(json.dumps(data, indent=2))


def cmd_rerank(stub: RerankerServiceStub, query: str, texts: list[str]) -> None:
    req = RerankRequest(query=query, texts=texts)
    resp = stub.Rerank(req)
    scores = list(resp.scores)
    print(json.dumps({"query": query, "texts": texts, "scores": scores}, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reranker Service CLI Client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8352)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Health check")

    p_rerank = sub.add_parser("rerank", help="Rerank candidate texts against a query")
    p_rerank.add_argument("query")
    p_rerank.add_argument("texts", nargs="+")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stub = create_stub(args.host, args.port)

    if args.command == "health":
        cmd_health(stub)
    elif args.command == "rerank":
        cmd_rerank(stub, args.query, args.texts)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
