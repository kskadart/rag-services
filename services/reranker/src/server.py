import logging
from concurrent import futures

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from grpc_health.v1.health import HealthServicer
from grpc_reflection.v1alpha import reflection

from src.config import settings
from src.reranker_pb2 import Empty, HealthResponse, RerankRequest, RerankResponse
from src.reranker_pb2_grpc import RerankerServiceServicer, add_RerankerServiceServicer_to_server
from src.service import get_reranker_service

logger = logging.getLogger(__name__)


class RerankerServicer(RerankerServiceServicer):
    """gRPC servicer for reranker service."""

    def __init__(self) -> None:
        """Initialize the servicer."""
        self.reranker_service = get_reranker_service()
        logger.info("Reranker servicer initialized")

    def Rerank(self, request: RerankRequest, context: grpc.ServicerContext) -> RerankResponse:
        """
        Score each candidate text against the query.

        Args:
            request: RerankRequest containing the query and candidate texts.
            context: gRPC context.

        Returns:
            RerankResponse containing scores in the same order as the input texts.
        """
        try:
            if not request.query or not request.query.strip():
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Query cannot be empty")
                return RerankResponse()

            if not request.texts:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Texts list cannot be empty")
                return RerankResponse()

            scores = self.reranker_service.rerank(query=request.query, texts=list(request.texts))

            return RerankResponse(scores=scores)

        except ValueError as e:
            logger.warning(f"Invalid request: {str(e)}")
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(str(e))
            return RerankResponse()
        except Exception as e:
            logger.error(f"Error in Rerank: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return RerankResponse()

    def HealthCheck(self, request: Empty, context: grpc.ServicerContext) -> HealthResponse:
        """
        Health check endpoint.

        Args:
            request: Empty request.
            context: gRPC context.

        Returns:
            HealthResponse with service status.
        """
        try:
            reranker_service = get_reranker_service()
            return HealthResponse(
                status="SERVING", model_name=reranker_service.model_name, device=reranker_service.device
            )

        except Exception as e:
            logger.error(f"Error in HealthCheck: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal server error: {str(e)}")
            return HealthResponse(status="NOT_SERVING")


def serve() -> None:
    """Start the gRPC server."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Starting reranker service on {settings.grpc_host}:{settings.grpc_port}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    servicer = RerankerServicer()
    servicer.reranker_service.load_model()
    add_RerankerServiceServicer_to_server(servicer, server)

    # Add health check service
    health_servicer = HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("reranker.RerankerService", health_pb2.HealthCheckResponse.SERVING)

    service_names = (
        "reranker.RerankerService",
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

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
