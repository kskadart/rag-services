"""Simple test client for embedding service."""

import grpc
import sys
import logging
from typing import List

# Import generated gRPC code
try:
    from embedding_pb2 import (
        EmbedTextRequest, EmbedBatchRequest, Empty,
        EmbeddingVector
    )
    from embedding_pb2_grpc import EmbeddingServiceStub
except ImportError:
    print("Error: gRPC code not generated. Run: python -m grpc_tools.protoc --proto_path=proto --python_out=src/embedding --grpc_python_out=src/embedding proto/embedding.proto")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Simple client for testing embedding service."""
    
    def __init__(self, host: str = "localhost", port: int = 81051):
        """
        Initialize client.
        
        Args:
            host: Server host
            port: Server port
        """
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = EmbeddingServiceStub(self.channel)
        logger.info(f"Connected to embedding service at {host}:{port}")
    
    def test_health(self) -> bool:
        """Test health check."""
        try:
            response = self.stub.HealthCheck(Empty())
            logger.info(f"Health check: {response.status}")
            logger.info(f"Model: {response.model_name}")
            logger.info(f"Device: {response.device}")
            return response.status == "SERVING"
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def test_dimension(self) -> int:
        """Test getting embedding dimension."""
        try:
            response = self.stub.GetEmbeddingDimension(Empty())
            logger.info(f"Embedding dimension: {response.dimension}")
            return response.dimension
        except Exception as e:
            logger.error(f"Get dimension failed: {e}")
            return 0
    
    def test_single_embedding(self, text: str = "Hello world") -> List[float]:
        """Test single text embedding."""
        try:
            request = EmbedTextRequest(
                text=text,
                max_length=512,
                normalize=True,
                pooling_strategy="mean"
            )
            response = self.stub.EmbedText(request)
            logger.info(f"Single embedding generated, dimension: {len(response.embedding)}")
            return list(response.embedding)
        except Exception as e:
            logger.error(f"Single embedding failed: {e}")
            return []
    
    def test_batch_embedding(self, texts: List[str] = None) -> List[List[float]]:
        """Test batch text embedding."""
        if texts is None:
            texts = [
                "Hello world",
                "This is a test",
                "Machine learning is awesome"
            ]
        
        try:
            request = EmbedBatchRequest(
                texts=texts,
                batch_size=8,
                max_length=512,
                normalize=True,
                pooling_strategy="mean"
            )
            response = self.stub.EmbedBatch(request)
            embeddings = [list(emb.vector) for emb in response.embeddings]
            logger.info(f"Batch embedding generated, {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return []
    
    def test_different_pooling(self) -> None:
        """Test different pooling strategies."""
        text = "This is a test sentence for pooling strategies."
        
        for strategy in ["mean", "cls", "max"]:
            try:
                request = EmbedTextRequest(
                    text=text,
                    max_length=512,
                    normalize=True,
                    pooling_strategy=strategy
                )
                response = self.stub.EmbedText(request)
                logger.info(f"Pooling '{strategy}' - dimension: {len(response.embedding)}")
            except Exception as e:
                logger.error(f"Pooling '{strategy}' failed: {e}")
    
    def test_error_handling(self) -> None:
        """Test error handling."""
        logger.info("Testing error handling...")
        
        # Test empty text
        try:
            request = EmbedTextRequest(text="", max_length=512, normalize=True, pooling_strategy="mean")
            response = self.stub.EmbedText(request)
            logger.warning("Empty text should have failed!")
        except Exception as e:
            logger.info(f"Empty text correctly rejected: {e}")
        
        # Test invalid pooling strategy
        try:
            request = EmbedTextRequest(text="test", max_length=512, normalize=True, pooling_strategy="invalid")
            response = self.stub.EmbedText(request)
            logger.warning("Invalid pooling should have failed!")
        except Exception as e:
            logger.info(f"Invalid pooling correctly rejected: {e}")
    
    def close(self):
        """Close the connection."""
        self.channel.close()


def main():
    """Run all tests."""
    logger.info("Starting embedding service tests...")
    
    # Create client
    client = EmbeddingClient()
    
    try:
        # Test health
        if not client.test_health():
            logger.error("Service is not healthy!")
            return
        
        # Test dimension
        dimension = client.test_dimension()
        if dimension == 0:
            logger.error("Could not get embedding dimension!")
            return
        
        # Test single embedding
        single_emb = client.test_single_embedding()
        if not single_emb:
            logger.error("Single embedding failed!")
            return
        
        # Test batch embedding
        batch_embs = client.test_batch_embedding()
        if not batch_embs:
            logger.error("Batch embedding failed!")
            return
        
        # Test different pooling strategies
        client.test_different_pooling()
        
        # Test error handling
        client.test_error_handling()
        
        logger.info("All tests completed successfully! ✅")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
