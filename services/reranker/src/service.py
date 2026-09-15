import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings

logger = logging.getLogger(__name__)

# transformers reports a very large sentinel value for tokenizer.model_max_length
# when the model config does not set an explicit limit; fall back to a sane default.
_UNSET_MAX_LENGTH_THRESHOLD = 1_000_000
_FALLBACK_MAX_LENGTH = 512
_BATCH_SIZE = 16


class RerankerService:
    """Cross-encoder reranker service scoring (query, text) pairs with a sequence classification model."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize reranker service.

        Args:
            model_name: Name of the cross-encoder reranker model.
        """
        self.model_name = model_name or settings.reranker_model_name
        self.model: AutoModelForSequenceClassification | None = None
        self.tokenizer: AutoTokenizer | None = None
        self.max_length: int = _FALLBACK_MAX_LENGTH

        if settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.device

        logger.info(f"Initializing reranker service with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")

    def load_model(self) -> None:
        """Load the reranker model and tokenizer."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=settings.hf_token,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=settings.hf_token,
            ).to(self.device)
        except Exception as e:
            # Provide clearer guidance when accessing gated/private repos
            message = str(e)
            if "gated" in message.lower() or "401" in message:
                logger.error(
                    "Hugging Face authentication required for model '%s'. "
                    "Set HF_TOKEN in environment or use a non-gated model.",
                    self.model_name,
                )
            raise

        self.model.eval()

        model_max_length = self.tokenizer.model_max_length
        self.max_length = model_max_length if model_max_length < _UNSET_MAX_LENGTH_THRESHOLD else _FALLBACK_MAX_LENGTH

        logger.info(f"Model loaded successfully. Max sequence length: {self.max_length}")

    def rerank(self, query: str, texts: list[str]) -> list[float]:
        """
        Score each candidate text against the query using the cross-encoder model.

        Args:
            query: Search query.
            texts: Candidate texts to score.

        Returns:
            Relevance scores (raw logits), one per input text, in the same order as `texts`.

        Raises:
            ValueError: If query is empty or texts list is empty.
            Exception: If reranking fails.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not texts:
            raise ValueError("Texts list cannot be empty")

        if self.model is None:
            self.load_model()
        assert self.model is not None
        assert self.tokenizer is not None

        try:
            scores: list[float] = []

            for i in range(0, len(texts), _BATCH_SIZE):
                batch = texts[i : i + _BATCH_SIZE]
                pairs = [[query, text] for text in batch]

                inputs = self.tokenizer(
                    pairs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

                with torch.no_grad():
                    logits = self.model(**inputs).logits.view(-1)

                scores.extend(logits.cpu().float().tolist())

            return scores

        except Exception as e:
            logger.error(f"Failed to rerank texts: {str(e)}")
            raise Exception(f"Failed to rerank texts: {str(e)}")


# Global reranker service instance
reranker_service: RerankerService | None = None


def get_reranker_service() -> RerankerService:
    """Get or create reranker service instance."""
    global reranker_service
    if reranker_service is None:
        reranker_service = RerankerService()
    return reranker_service
