# Reranker Service

gRPC-based cross-encoder reranker service using BAAI's bge-reranker-v2-m3 model.

## Features

- **Cross-Encoder Scoring**: Scores each (query, text) pair jointly for higher relevance precision than embedding similarity
- **CPU & GPU Support**: Separate Docker images for different hardware
- **Batched Inference**: Internally batches candidate texts for throughput
- **Health Checks**: Built-in health monitoring
- **gRPC Reflection**: Easy debugging and testing

## Quick Start

### CPU Deployment (Development)

```bash
# Start the service
docker compose -f compose.cpu.yml up -d --build reranker-service

# Test the service
cd services/reranker
python client_cli.py health
python client_cli.py rerank "What is the capital of France?" "Paris is the capital of France." "Bananas are yellow."
```

## API Reference

### gRPC Service: `reranker.RerankerService`

#### Methods

**Rerank**
```protobuf
rpc Rerank(RerankRequest) returns (RerankResponse);
```

Scores each text against the query. `RerankResponse.scores` are returned in the same order as the input `texts`; a higher score means more relevant.

**HealthCheck**
```protobuf
rpc HealthCheck(Empty) returns (HealthResponse);
```

### Example Usage

```python
import grpc
from reranker_pb2 import RerankRequest
from reranker_pb2_grpc import RerankerServiceStub

channel = grpc.insecure_channel('localhost:8352')
stub = RerankerServiceStub(channel)

request = RerankRequest(
    query="What is the capital of France?",
    texts=[
        "Paris is the capital of France.",
        "Bananas are a good source of potassium.",
    ],
)
response = stub.Rerank(request)
scores = list(response.scores)
```

## Configuration

### Environment Variables

- `RERANKER_MODEL_NAME`: Model name (default: BAAI/bge-reranker-v2-m3)
- `HF_TOKEN`: Optional; only required for gated/private Hugging Face models
- `DEVICE`: Device selection (auto/cuda/cpu)
- `GRPC_PORT`: Server port (default: 8352)
- `GRPC_HOST`: Server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Information

- **Model**: BAAI/bge-reranker-v2-m3
- **Architecture**: Cross-encoder sequence classification (XLM-RoBERTa based)
- **Device**: CUDA (GPU) or CPU

## Testing

### Using grpcurl

```bash
# List services
grpcurl -plaintext localhost:8352 list

# Health check
grpcurl -plaintext localhost:8352 grpc.health.v1.Health/Check

# Rerank
grpcurl -plaintext -d '{"query": "What is the capital of France?", "texts": ["Paris is the capital of France.", "Bananas are yellow."]}' \
  localhost:8352 reranker.RerankerService/Rerank
```

### Using Python Test Client

```bash
cd services/reranker
python client_cli.py rerank "What is the capital of France?" "Paris is the capital of France." "Bananas are yellow."
```

## Development

### Setup

```bash
cd services/reranker

# Install dependencies
uv sync --extra cpu

# Generate gRPC code
uv run python -m grpc_tools.protoc \
  -Isrc=proto \
  --python_out=. \
  --grpc_python_out=. \
  proto/reranker.proto

# Run server
uv run python -m src.server
```

### Building Docker Images

```bash
# CPU image
docker build -f docker/Dockerfile.cpu -t reranker-service:cpu .
```

## Error Handling

The service raises gRPC errors for:
- Empty query
- Empty texts list
- Model loading failures

Use appropriate gRPC status codes for error handling in clients.
