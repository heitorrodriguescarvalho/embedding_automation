"""Persist embedding records to Parquet files."""

from typing import List

import pyarrow as pa
import pyarrow.parquet as pq

_COLUMNS = [
    "lattes_id",
    "chunk_index",
    "content",
    "metadata_filename",
    "metadata_timestamp",
    "embedding",
]


def _embedding_dims(embedding) -> int:
    return len(embedding)


def save_shard_parquet(records: List[dict], output_path: str) -> None:
    if not records:
        raise ValueError("cannot write an empty shard (no records)")

    first_dims = _embedding_dims(records[0]["embedding"])
    for i, record in enumerate(records):
        dims = _embedding_dims(record["embedding"])
        if dims != first_dims:
            raise ValueError(
                f"inconsistent embedding dimensions: record 0 has {first_dims} "
                f"dims but record {i} (lattes_id={record.get('lattes_id')!r}) "
                f"has {dims} dims"
            )

    schema = pa.schema([
        pa.field("lattes_id", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field("content", pa.string()),
        pa.field("metadata_filename", pa.string()),
        pa.field("metadata_timestamp", pa.string()),
        pa.field(
            "embedding",
            pa.list_(pa.float32()),
        ),
    ])

    data = {
        column: [record[column] for record in records]
        for column in _COLUMNS
    }
    table = pa.Table.from_pydict(data, schema=schema)

    pq.write_table(table, output_path, compression="zstd")
