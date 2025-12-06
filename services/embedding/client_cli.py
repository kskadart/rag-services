import argparse
import json
import logging
from typing import List

import grpc

from tests.embedding_pb2 import (
    EmbedTextRequest,
    EmbedBatchRequest,
    Empty,
)
from tests.embedding_pb2_grpc import EmbeddingServiceStub


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_stub(host: str, port: int) -> EmbeddingServiceStub:
    channel = grpc.insecure_channel(f"{host}:{port}")
    return EmbeddingServiceStub(channel)


def cmd_health(stub: EmbeddingServiceStub) -> None:
    resp = stub.HealthCheck(Empty())
    data = {"status": resp.status, "model_name": resp.model_name, "device": resp.device}
    print(json.dumps(data, indent=2))


def cmd_dimension(stub: EmbeddingServiceStub) -> None:
    resp = stub.GetEmbeddingDimension(Empty())
    print(resp.dimension)


def cmd_embed_text(stub: EmbeddingServiceStub, text: str, max_length: int, normalize: bool, pooling: str) -> None:
    req = EmbedTextRequest(text=text, max_length=max_length, normalize=normalize, pooling_strategy=pooling)
    resp = stub.EmbedText(req)
    print(json.dumps(list(resp.embedding)))


def cmd_embed_batch(
    stub: EmbeddingServiceStub,
    texts: List[str],
    batch_size: int,
    max_length: int,
    normalize: bool,
    pooling: str,
) -> None:
    req = EmbedBatchRequest(
        texts=texts,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
        pooling_strategy=pooling,
    )
    resp = stub.EmbedBatch(req)
    embeddings = [list(emb.vector) for emb in resp.embeddings]
    print(json.dumps(embeddings))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embedding Service CLI Client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8351)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="Health check")
    sub.add_parser("dimension", help="Get embedding dimension")

    p_text = sub.add_parser("embed-text", help="Embed a single text")
    p_text.add_argument("text")
    p_text.add_argument("--max-length", type=int, default=512)
    p_text.add_argument("--normalize", action="store_true")
    p_text.add_argument("--pooling", choices=["mean", "cls", "max"], default="mean")

    p_batch = sub.add_parser("embed-batch", help="Embed a batch of texts")
    p_batch.add_argument("texts", nargs="+")
    p_batch.add_argument("--batch-size", type=int, default=8)
    p_batch.add_argument("--max-length", type=int, default=512)
    p_batch.add_argument("--normalize", action="store_true")
    p_batch.add_argument("--pooling", choices=["mean", "cls", "max"], default="mean")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stub = create_stub(args.host, args.port)

    if args.command == "health":
        cmd_health(stub)
    elif args.command == "dimension":
        cmd_dimension(stub)
    elif args.command == "embed-text":
        cmd_embed_text(stub, args.text, args.max_length, args.normalize, args.pooling)
    elif args.command == "embed-batch":
        cmd_embed_batch(stub, args.texts, args.batch_size, args.max_length, args.normalize, args.pooling)
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()


