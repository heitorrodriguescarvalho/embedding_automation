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
    batch_meta: list | None = None,
) -> tuple[np.ndarray, list[int]]:
    """Embed texts, returning (embeddings, skipped_indices).

    Embeddings are aligned with the kept texts (input order minus skipped).
    Texts that still hit OutOfMemoryError with batch_size=1 are skipped and
    reported on the terminal instead of stopping the pipeline.
    batch_meta provides an optional parallel list of labels (e.g. source
    filenames) used in the skip messages.
    """
    with torch.inference_mode():
        embeddings, skipped = _encode_with_oom_skips(
            model,
            texts,
            batch_size,
            batch_meta,
        )
    return np.asarray(embeddings, dtype=np.float32), skipped


def _encode_with_oom_skips(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    batch_meta: list | None,
) -> tuple[np.ndarray, list[int]]:
    """Encode texts in batches, halving the batch size on OOM until it fits.

    When a batch still OOMs with a single text, that text is skipped and
    reported, and the remaining texts keep being embedded.
    """
    if batch_meta is None:
        batch_meta = [None] * len(texts)

    out: list[np.ndarray] = []
    skipped: list[int] = []
    n = len(texts)
    i = 0

    while i < n:
        end = min(i + batch_size, n)
        embeddings = _try_encode_batch(model, texts[i:end], batch_size)
        if embeddings is not None:
            out.append(embeddings)
            i = end
            continue

        while i < end:
            single = _try_encode_batch(model, texts[i:i + 1], 1)
            if single is not None:
                out.append(single)
            else:
                skipped.append(i)
                _report_skipped(batch_meta[i], i)
            i += 1

    if skipped:
        print(
            f"[embedder] skipped {len(skipped)} text(s) that OOM'd even "
            f"alone; continuing without them"
        )

    if not out:
        raise RuntimeError("embedding failed: every batch hit OutOfMemoryError")

    return np.concatenate(out, axis=0), skipped


def _try_encode_batch(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
) -> np.ndarray | None:
    """Encode a batch, retrying with a smaller batch size on OOM.

    Returns None when even a single text keeps running out of memory.
    """
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
                return None
            original_batch_size = batch_size
            batch_size = max(1, batch_size - 8)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"[embedder] OutOfMemoryError: reducing batch_size "
                f"{original_batch_size} -> {batch_size} and retrying"
            )


def _report_skipped(label, index: int) -> None:
    label = label if label is not None else f"index {index}"
    print(
        f"[embedder] WARNING: skipping text at index {index} "
        f"(file: {label}) - OutOfMemoryError even at batch_size=1"
    )


def _is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return "out of memory" in str(exc).lower()
    return False


def get_tokenizer(model: SentenceTransformer):
    return model.tokenizer
