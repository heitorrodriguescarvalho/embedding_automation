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
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    return np.asarray(embeddings, dtype=np.float32)


def get_tokenizer(model: SentenceTransformer):
    return model.tokenizer
