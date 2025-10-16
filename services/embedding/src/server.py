import grpc
import logging
from concurrent import futures

from grpc_health.v1 import health_pb2_grpc
from grpc_health.v1 import health_pb2
from grpc_health.v1.health import HealthServicer
from grpc_reflection.v1alpha import reflection

from src.embedding_pb2 import (
    EmbedTextRequest,
    EmbedTextResponse,
    EmbedBatchRequest,
    EmbedBatchResponse,
    EmbeddingVector,
    Empty,
    DimensionResponse,
    HealthResponse,
)
from src.embedding_pb2_grpc import EmbeddingServiceServicer, add_EmbeddingServiceServicer_to_server
from src.service import get_embedding_service
from src.config import settings

logger = logging.getLogger(__name__)


class EmbeddingServicer(EmbeddingServiceServicer):
    """gRPC servicer for embedding service."""

    def __init__(self):
        """Initialize the servicer."""
        self.embedding_service = get_embedding_service()
        logger.info("Embedding servicer initialized")

    def EmbedText(self, request: EmbedTextRequest, context) -> EmbedTextResponse:
        """
        Generate embedding for a single text.

        Args:
            request: EmbedTextRequest containing text and parameters
            context: gRPC context

        Returns:
            EmbedTextResponse containing the embedding vector
        """
        try:
            # Validate request
            if not request.text or not request.text.strip():
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Text cannot be empty")
                return EmbedTextResponse()

            # Generate embedding
            embedding = self.embedding_service.embed_text(
                text=request.text,
                max_length=request.max_length,
                normalize=request.normalize,
                pooling_strategy=request.pooling_strategy,
            )

            return EmbedTextResponse(embedding=embedding)

        except ValueError as e:
            logger.warning(f"Invalid request: {str(e)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return EmbedTextResponse()
        except Exception as e:
            logger.error(f"Error in EmbedText: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return EmbedTextResponse()

    def EmbedBatch(self, request: EmbedBatchRequest, context) -> EmbedBatchResponse:
        """
        Generate embeddings for multiple texts.

        Args:
            request: EmbedBatchRequest containing texts and parameters
            context: gRPC context

        Returns:
            EmbedBatchResponse containing embedding vectors
        """
        try:
            # Validate request
            if not request.texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Texts list cannot be empty")
                return EmbedBatchResponse()

            # Generate embeddings
            embeddings = self.embedding_service.embed_batch(
                texts=list(request.texts),
                batch_size=request.batch_size,
                max_length=request.max_length,
                normalize=request.normalize,
                pooling_strategy=request.pooling_strategy,
            )

            # Convert to EmbeddingVector objects
            embedding_vectors = [EmbeddingVector(vector=embedding) for embedding in embeddings]

            return EmbedBatchResponse(embeddings=embedding_vectors)

        except ValueError as e:
            logger.warning(f"Invalid request: {str(e)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return EmbedBatchResponse()
        except Exception as e:
            logger.error(f"Error in EmbedBatch: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return EmbedBatchResponse()

    def GetEmbeddingDimension(self, request: Empty, context) -> DimensionResponse:
        """
        Get the dimension of embeddings produced by the model.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            DimensionResponse containing the embedding dimension
        """
        try:
            dimension = self.embedding_service.get_embedding_dimension()
            return DimensionResponse(dimension=dimension)

        except Exception as e:
            logger.error(f"Error in GetEmbeddingDimension: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return DimensionResponse()

    def HealthCheck(self, request: Empty, context) -> HealthResponse:
        """
        Health check endpoint.

        Args:
            request: Empty request
            context: gRPC context

        Returns:
            HealthResponse with service status
        """
        try:
            embedding_service = get_embedding_service()
            return HealthResponse(
                status="SERVING", model_name=embedding_service.model_name, device=embedding_service.device
            )

        except Exception as e:
            logger.error(f"Error in HealthCheck: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return HealthResponse(status="NOT_SERVING")


def serve():
    """Start the gRPC server."""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Starting embedding service on {settings.grpc_host}:{settings.grpc_port}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    servicer = EmbeddingServicer()
    servicer.embedding_service.load_model()
    add_EmbeddingServiceServicer_to_server(servicer, server)

    # Add health check service
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("embedding.EmbeddingService", health_pb2.HealthCheckResponse.SERVING)

    service_names = (
        "embedding.EmbeddingService",
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    # Start server
    listen_addr = f"{settings.grpc_host}:{settings.grpc_port}"
    server.add_insecure_port(listen_addr)
    server.start()

    logger.info(f"Server started, listening on {listen_addr}")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(grace=5)


if __name__ == "__main__":
    serve()
