"""Wrapper for easy cross-encoder inference."""

import numpy as np
import torch

from .modeling_swipe import SwipeCrossEncoderForSequenceClassification
from .processing_swipe import SwipeProcessor
from .tokenization_swipe import SwipeTokenizer


class SwipeCrossEncoder:
    """
    Wrapper around SwipeCrossEncoderForSequenceClassification for easy inference.

    This class provides a simple interface for using the swipe cross-encoder model,
    handling all the preprocessing and postprocessing automatically.

    Note: This is a custom wrapper, not compatible with sentence-transformers.CrossEncoder
    due to the multimodal (path + text) nature of the inputs.

    Usage:
        ```python
        from swipealot.huggingface import SwipeCrossEncoder

        # Load model
        model = SwipeCrossEncoder("path/to/model")

        # Swipe path coordinates
        path = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1], [0.2, 0.3, 0.2]]

        # Candidate words
        words = ["hello", "world", "help", "hold"]

        # Get similarity scores
        scores = model.predict(path, words)

        # Rank candidates
        ranked = model.rank(path, words, top_k=3)
        ```

    Args:
        model_name_or_path (str): Path to model directory or HuggingFace Hub model ID
        device (str, optional): Device to run inference on. If None, uses CUDA if available.
        **kwargs: Additional arguments passed to from_pretrained()
    """

    def __init__(self, model_name_or_path: str, device: str = None, **kwargs):
        """
        Initialize the cross-encoder wrapper.

        Args:
            model_name_or_path: Path to model directory or HuggingFace Hub model ID
            device: Device to run inference on ("cpu", "cuda", etc.)
            **kwargs: Additional arguments for from_pretrained()
        """
        # Load model
        self.model = SwipeCrossEncoderForSequenceClassification.from_pretrained(
            model_name_or_path, **kwargs
        )

        # Load tokenizer and processor
        self.tokenizer = SwipeTokenizer.from_pretrained(model_name_or_path)

        # Get max lengths from config
        max_path_len = getattr(self.model.config, "max_path_len", 64)
        max_char_len = getattr(self.model.config, "max_char_len", 38)

        self.processor = SwipeProcessor(
            tokenizer=self.tokenizer,
            max_path_len=max_path_len,
            max_char_len=max_char_len,
        )

        # Set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        print(f"Loaded SwipeCrossEncoder on {self.device}")

    def predict(
        self,
        path_coords: list[list[list[float]]] | torch.Tensor | np.ndarray,
        words: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """
        Predict similarity scores for path-word pairs.

        Args:
            path_coords: Swipe path coordinates. Can be:
                - Single path: [[x,y,t], [x,y,t], ...] shape [path_len, 3]
                - Batch of paths: [path1, path2, ...] shape [batch, path_len, 3]
                When single path is provided with multiple words, it will be
                repeated for each word.
            words: Single word (str) or list of words (List[str])
            batch_size: Batch size for processing (not used yet, for future optimization)
            show_progress_bar: Whether to show progress bar (not used yet)

        Returns:
            np.ndarray: Similarity scores. Shape depends on inputs:
                - If single path + single word: scalar
                - If single path + N words: [N]
                - If N paths + N words: [N] (pairwise)
        """
        # Ensure words is a list
        if isinstance(words, str):
            words = [words]
            single_word = True
        else:
            single_word = False

        # Handle path_coords
        if isinstance(path_coords, (list, tuple)):
            # Check if it's a single path or batch
            if len(path_coords) > 0 and not isinstance(path_coords[0][0], (list, tuple)):
                # Single path [[x,y,t], ...] - repeat for each word
                path_coords = [path_coords] * len(words)
        elif isinstance(path_coords, (torch.Tensor, np.ndarray)):
            if isinstance(path_coords, np.ndarray):
                path_coords = torch.from_numpy(path_coords)

            if path_coords.dim() == 2:
                # Single path [path_len, 3] - repeat for each word
                path_coords = path_coords.unsqueeze(0).repeat(len(words), 1, 1)
            elif path_coords.shape[0] == 1 and len(words) > 1:
                # Single path in batch format - repeat
                path_coords = path_coords.repeat(len(words), 1, 1)

        # Prepare inputs
        inputs = self.processor(path_coords, words, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1)

        # Convert to numpy
        scores_np = scores.cpu().numpy()

        # Return scalar if single word input
        if single_word:
            return scores_np[0] if scores_np.shape[0] == 1 else scores_np

        return scores_np

    def rank(
        self,
        path_coords: list[list[list[float]]] | torch.Tensor | np.ndarray,
        candidates: list[str],
        top_k: int = None,
        return_scores: bool = True,
    ) -> list[tuple]:
        """
        Rank candidate words for a single swipe path.

        Args:
            path_coords: Single swipe path [[x,y,t], ...] shape [path_len, 3]
            candidates: List of candidate words to rank
            top_k: Return only top K results. If None, returns all.
            return_scores: Whether to return (word, score) tuples or just words

        Returns:
            List of (word, score) tuples sorted by score descending,
            or List of words if return_scores=False
        """
        # Get scores for all candidates
        scores = self.predict(path_coords, candidates)

        # Create (word, score) pairs
        ranked = list(zip(candidates, scores, strict=False))

        # Sort by score descending
        ranked.sort(key=lambda x: x[1], reverse=True)

        # Limit to top_k
        if top_k is not None:
            ranked = ranked[:top_k]

        # Return words only if requested
        if not return_scores:
            return [word for word, _ in ranked]

        return ranked

    def batch_predict(
        self,
        path_coords_batch: list[list[list[list[float]]]],
        words_batch: list[list[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> list[np.ndarray]:
        """
        Predict scores for multiple path-word groups.

        Args:
            path_coords_batch: List of path coordinate lists
            words_batch: List of word lists (must match length of path_coords_batch)
            batch_size: Processing batch size
            show_progress_bar: Whether to show progress

        Returns:
            List of score arrays, one per group
        """
        if len(path_coords_batch) != len(words_batch):
            raise ValueError(
                f"Length mismatch: {len(path_coords_batch)} paths vs {len(words_batch)} word groups"
            )

        results = []
        for path, words in zip(path_coords_batch, words_batch, strict=False):
            scores = self.predict(path, words, batch_size=batch_size, show_progress_bar=False)
            results.append(scores)

        return results

    def encode(
        self,
        path_coords: list[list[list[float]]] | torch.Tensor | np.ndarray,
        words: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """
        Encode path-word pairs into embeddings (SEP token representations).

        This method extracts the SEP token embedding, which represents the
        joint encoding of the path and word. These embeddings can be used for:
        - Vector databases (Pinecone, Weaviate, Qdrant, etc.)
        - Semantic search
        - Clustering
        - Similarity computation

        Args:
            path_coords: Swipe path coordinates
            words: Word or list of words
            batch_size: Processing batch size (not used yet)
            show_progress_bar: Whether to show progress (not used yet)
            convert_to_numpy: Whether to convert to numpy array
            normalize_embeddings: Whether to L2-normalize embeddings

        Returns:
            Embeddings of shape [batch, d_model] (numpy array or torch tensor)

        Example:
            ```python
            # Single embedding
            embedding = encoder.encode(path, "hello")
            # Shape: [d_model]

            # Batch embeddings
            embeddings = encoder.encode([path1, path2], ["hello", "world"])
            # Shape: [2, d_model]

            # For vector database
            import chromadb
            client = chromadb.Client()
            collection = client.create_collection("swipe_embeddings")

            embedding = encoder.encode(path, "hello")
            collection.add(
                embeddings=[embedding.tolist()],
                documents=["hello"],
                ids=["1"]
            )
            ```
        """
        # Ensure words is a list
        if isinstance(words, str):
            words = [words]
            single_input = True
        else:
            single_input = False

        # Handle path_coords batching (same as predict)
        if isinstance(path_coords, (list, tuple)):
            if len(path_coords) > 0 and not isinstance(path_coords[0][0], (list, tuple)):
                # Single path - repeat for each word
                path_coords = [path_coords] * len(words)
        elif isinstance(path_coords, (torch.Tensor, np.ndarray)):
            if isinstance(path_coords, np.ndarray):
                path_coords = torch.from_numpy(path_coords)
            if path_coords.dim() == 2:
                path_coords = path_coords.unsqueeze(0).repeat(len(words), 1, 1)
            elif path_coords.shape[0] == 1 and len(words) > 1:
                path_coords = path_coords.repeat(len(words), 1, 1)

        # Prepare inputs
        inputs = self.processor(path_coords, words, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract SEP token embeddings
        with torch.no_grad():
            # Get hidden states from the encoder
            embeddings_input = self.model.embeddings(
                inputs["path_coords"],
                inputs["input_ids"],
                torch.full(
                    (inputs["path_coords"].shape[0], 1),
                    fill_value=self.model.config.cls_token_id,
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.full(
                    (inputs["path_coords"].shape[0], 1),
                    fill_value=self.model.config.sep_token_id,
                    dtype=torch.long,
                    device=self.device,
                ),
            )

            # Encode
            src_key_padding_mask = (
                inputs["attention_mask"] == 0 if "attention_mask" in inputs else None
            )
            hidden_states = self.model.encoder(
                embeddings_input, src_key_padding_mask=src_key_padding_mask
            )

            # Extract SEP token embedding
            path_len = inputs["path_coords"].shape[1]
            sep_position = 1 + path_len
            embeddings = hidden_states[:, sep_position, :]  # [batch, d_model]

        # Normalize if requested
        if normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to numpy if requested
        if convert_to_numpy:
            embeddings = embeddings.cpu().numpy()

        # Return scalar if single input
        if single_input:
            return embeddings[0] if embeddings.shape[0] == 1 else embeddings

        return embeddings

    def __call__(self, path_coords, words, **kwargs):
        """
        Shorthand for predict().

        Args:
            path_coords: Swipe path coordinates
            words: Word or list of words
            **kwargs: Additional arguments for predict()

        Returns:
            Similarity scores
        """
        return self.predict(path_coords, words, **kwargs)
