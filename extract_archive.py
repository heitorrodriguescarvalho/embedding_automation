"""Streaming extraction of .tar.zst archives containing Lattes curricula .md files."""

import logging
import os
import sys
import tarfile
from typing import Iterator

import zstandard

logger = logging.getLogger(__name__)


def iter_curriculum_files(archive_path: str) -> Iterator[dict]:
    with open(archive_path, "rb") as f:
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not member.isreg():
                        continue
                    if not member.name.endswith(".md"):
                        continue

                    filename = os.path.basename(member.name)
                    lattes_id = filename[:-3]
                    try:
                        content = tar.extractfile(member)
                    except Exception as e:
                        logger.warning("failed to read %s: %s", filename, e)
                        continue
                    if content is None:
                        continue

                    try:
                        raw = content.read()
                        text = raw.decode("utf-8", errors="replace")
                    except Exception as e:
                        logger.warning("failed to decode %s: %s", filename, e)
                        continue

                    yield {
                        "lattes_id": lattes_id,
                        "filename": filename,
                        "content": text,
                    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python extract_archive.py <archive.tar.zst>")
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING)

    archive = sys.argv[1]
    count = 0
    for record in iter_curriculum_files(archive):
        print(record["lattes_id"])
        count += 1
        if count >= 3:
            break
