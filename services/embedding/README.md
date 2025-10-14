# Embedding Service

High-performance gRPC-based embedding service using Google's EmbeddingGemma model.

## Features

- **Single & Batch Processing**: Embed individual texts or batches efficiently
- **GPU & CPU Support**: Separate Docker images for different hardware
- **Client-Controlled Parameters**: Full control over processing parameters per request
- **Multiple Pooling Strategies**: Mean, CLS, and Max pooling
- **Health Checks**: Built-in health monitoring
- **gRPC Reflection**: Easy debugging and testing

## Quick Start

### CPU Deployment (Development)

```bash
# Start the service
docker-compose -f compose.cpu.yml up

# Test the service
cd services/embedding
python tests/test_client.py
```

### GPU Deployment (Production)

```bash
# Start the service
docker-compose -f compose.gpu.yml up

# Test the service
cd services/embedding
python tests/test_client.py
```

## API Reference

### gRPC Service: `embedding.EmbeddingService`

#### Methods

**EmbedText**
```protobuf
rpc EmbedText(EmbedTextRequest) returns (EmbedTextResponse);
```

**EmbedBatch**
```protobuf
rpc EmbedBatch(EmbedBatchRequest) returns (EmbedBatchResponse);
```

**GetEmbeddingDimension**
```protobuf
rpc GetEmbeddingDimension(Empty) returns (DimensionResponse);
```

**HealthCheck**
```protobuf
rpc HealthCheck(Empty) returns (HealthResponse);
```

### Request Parameters

All processing parameters are **client-controlled**:

- `max_length` (int): Maximum token length (raises error if exceeded)
- `batch_size` (int): Batch size for processing
- `normalize` (bool): Whether to L2-normalize embeddings
- `pooling_strategy` (string): "mean", "cls", or "max"

### Example Usage

```python
import grpc
from embedding_pb2 import EmbedBatchRequest
from embedding_pb2_grpc import EmbeddingServiceStub

# Connect to service
channel = grpc.insecure_channel('localhost:81051')
stub = EmbeddingServiceStub(channel)

# Embed single text
request = EmbedTextRequest(
    text="Hello world",
    max_length=512,
    normalize=True,
    pooling_strategy="mean"
)
response = stub.EmbedText(request)
embedding = list(response.embedding)

# Embed batch
request = EmbedBatchRequest(
    texts=["Text 1", "Text 2", "Text 3"],
    batch_size=8,
    max_length=512,
    normalize=True,
    pooling_strategy="mean"
)
response = stub.EmbedBatch(request)
embeddings = [list(emb.vector) for emb in response.embeddings]
```

## Configuration

### Environment Variables

- `EMBEDDING_MODEL_NAME`: Model name (default: google/embeddinggemma-300m)
- `DEVICE`: Device selection (auto/cuda/cpu)
- `GRPC_PORT`: Server port (default: 81051)
- `GRPC_HOST`: Server host (default: 0.0.0.0)
- `LOG_LEVEL`: Logging level (default: INFO)

### Model Information

- **Model**: google/embeddinggemma-300m
- **Dimension**: 256
- **Max Context**: 512 tokens
- **Device**: CUDA (GPU) or CPU

## Testing

### Using grpcurl

```bash
# List services
grpcurl -plaintext localhost:81051 list

# Health check
grpcurl -plaintext localhost:81051 grpc.health.v1.Health/Check

# Single embedding
grpcurl -plaintext -d '{"text": "hello world", "max_length": 512, "normalize": true, "pooling_strategy": "mean"}' \
  localhost:81051 embedding.EmbeddingService/EmbedText
```

### Using Python Test Client

```bash
cd services/embedding
python tests/test_client.py
```

## Development

### Setup

```bash
cd services/embedding

# Install dependencies
uv pip install -e .

# Generate gRPC code
python -m grpc_tools.protoc \
  --proto_path=proto \
  --python_out=src/embedding \
  --grpc_python_out=src/embedding \
  proto/embedding.proto

# Run server
python -m src.embedding.server
```

### Building Docker Images

```bash
# CPU image
docker build -f docker/Dockerfile.cpu -t embedding-service:cpu .

# GPU image
docker build -f docker/Dockerfile.gpu -t embedding-service:gpu .
```

## Performance

### CPU Performance
- Single text: ~200ms
- Batch of 8: ~1.5s
- Memory: ~2GB

### GPU Performance (T4)
- Single text: ~10ms
- Batch of 8: ~30ms
- Memory: ~4GB

## Error Handling

The service raises gRPC errors for:
- Empty text input
- Text exceeding max_length
- Invalid pooling strategies
- Model loading failures

Use appropriate gRPC status codes for error handling in clients.
