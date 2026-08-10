# Contratos de interface 

```
extract_archive.py
  iter_curriculum_files(archive_path: str) -> Iterator[dict]
    yield {"lattes_id": str, "filename": str, "content": str}

chunker.py
  chunk_markdown(text: str, tokenizer, max_tokens: int) -> list[dict]
    return [{"secao": str, "content": str, "n_tokens": int}, ...]
    (lista já ordenada; quem chama define o chunk_index)

checkpoint.py
  load_checkpoint(path: str) -> set[str]          # ids já processados
  save_checkpoint(path: str, processed_ids: set)  # escrita atômica

embedder.py
  load_embedding_model(model_name: str, device: str, load_in_8bit: bool) -> SentenceTransformer
  embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray  # float32, shape (N, dim)

save_parquet.py
  save_shard_parquet(records: list[dict], output_path: str) -> None
    records: [{"lattes_id", "chunk_index", "content",
               "metadata_filename", "metadata_timestamp", "embedding"}, ...]
```
