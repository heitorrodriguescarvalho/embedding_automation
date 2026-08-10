"""Embedding model loading and batch encoding."""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_embedding_model(
    model_name: str,
    device: str = "cuda",
) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16,
            "attn_implementation": "sdpa",
        },
        processor_kwargs={"padding_side": "left"},
    )

    return model


def embed_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = 32,
) -> np.ndarray:
    with torch.inference_mode():
        embeddings = _encode_with_oom_retry(
            model,
            texts,
            batch_size,
        )
    return np.asarray(embeddings, dtype=np.float32)


def _encode_with_oom_retry(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray:
    """Encode texts, halving the batch size on OutOfMemoryError until it fits.

    After a successful run, the original batch size is restored so subsequent
    calls start from the caller-provided value again.
    """
    original_batch_size = batch_size

    while True:
        try:
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_oom(exc):
                raise
            if batch_size <= 1:
                raise
            batch_size = max(1, batch_size - 8)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[embedder] OutOfMemoryError: reducing batch_size "
                f"{original_batch_size} -> {batch_size} and retrying"
            )


def _is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def get_tokenizer(model: SentenceTransformer):
    return model.tokenizer
