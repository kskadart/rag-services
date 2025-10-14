"""Core embedding service implementation."""

from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import logging

from .config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using transformer models."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the model for embeddings
        """
        self.model_name = model_name or settings.embedding_model_name
        self.model: Optional[AutoModel] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.embedding_dimension: Optional[int] = None
        
        # Determine device
        if settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.device
            
        logger.info(f"Initializing embedding service with model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self) -> None:
        """Load the embedding model."""
        if self.model is None:
            logger.info(f"Loading model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self.model.eval()
            
            # Get embedding dimension
            with torch.no_grad():
                test_inputs = self.tokenizer("test", return_tensors="pt", padding=True, truncation=True)
                test_inputs = {k: v.to(self.device) for k, v in test_inputs.items()}
                test_output = self.model(**test_inputs)
                # Use mean pooling of last hidden state
                self.embedding_dimension = test_output.last_hidden_state.shape[-1]
                
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _cls_pooling(self, model_output):
        """Apply CLS pooling (use first token)."""
        return model_output.last_hidden_state[:, 0]
    
    def _max_pooling(self, model_output, attention_mask):
        """Apply max pooling."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set masked tokens to large negative value
        return torch.max(token_embeddings, 1)[0]
    
    def embed_text(
        self, 
        text: str, 
        max_length: int,
        normalize: bool,
        pooling_strategy: str
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            max_length: Maximum token length
            normalize: Whether to normalize embeddings
            pooling_strategy: Pooling strategy (mean/cls/max)
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty or too long
            Exception: If embedding fails
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        
        if self.model is None:
            self.load_model()
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Check if text exceeds max_length
            if len(inputs['input_ids'][0]) > max_length:
                raise ValueError(f"Text exceeds maximum length of {max_length} tokens")
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Apply pooling strategy
                if pooling_strategy == "mean":
                    embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                elif pooling_strategy == "cls":
                    embeddings = self._cls_pooling(outputs)
                elif pooling_strategy == "max":
                    embeddings = self._max_pooling(outputs, inputs['attention_mask'])
                else:
                    raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                
                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()[0].tolist()
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Failed to generate embedding: {str(e)}")
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int,
        max_length: int,
        normalize: bool,
        pooling_strategy: str
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            max_length: Maximum token length
            normalize: Whether to normalize embeddings
            pooling_strategy: Pooling strategy (mean/cls/max)
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts list is empty or contains invalid texts
            Exception: If embedding fails
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        if self.model is None:
            self.load_model()
        
        try:
            all_embeddings = []
            
            # Process in batches
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i + batch_size]
                
                # Tokenize batch
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                )
                
                # Check if any text exceeds max_length
                for j, input_ids in enumerate(inputs['input_ids']):
                    if len(input_ids) > max_length:
                        raise ValueError(f"Text at index {i+j} exceeds maximum length of {max_length} tokens")
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Apply pooling strategy
                    if pooling_strategy == "mean":
                        embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                    elif pooling_strategy == "cls":
                        embeddings = self._cls_pooling(outputs)
                    elif pooling_strategy == "max":
                        embeddings = self._max_pooling(outputs, inputs['attention_mask'])
                    else:
                        raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
                    
                    # Normalize if requested
                    if normalize:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
            
            return all_embeddings
        
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise Exception(f"Failed to generate embeddings: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            self.load_model()
        return self.embedding_dimension


# Global embedding service instance
embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service instance."""
    global embedding_service
    if embedding_service is None:
        embedding_service = EmbeddingService()
    return embedding_service
