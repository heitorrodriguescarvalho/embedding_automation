"""Markdown chunking for Lattes curricula."""

from typing import Iterator, Optional


def _iter_sections(text: str) -> Iterator[tuple[str, str]]:
    """Yield (section_path, raw_body) for each header-delimited section.

    The section path is the full "Parent > Child" chain of the header.
    """
    lines = text.splitlines()
    path: list[tuple[int, str]] = []
    current_body: list[str] = []

    def current_path() -> str:
        return " > ".join(title for _, title in path)

    def flush() -> Optional[tuple[str, str]]:
        nonlocal current_body
        if not path:
            current_body = []
            return None
        section = (current_path(), "\n".join(current_body).strip())
        current_body = []
        return section

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            if path:
                section = flush()
                if section is not None:
                    yield section

            header = stripped
            title = ""
            level = 0
            for lvl in (1, 2, 3):
                prefix = "#" * lvl + " "
                if header.startswith(prefix):
                    title = header[len(prefix):].strip()
                    level = lvl
                    break
            if title:
                path = [(l, t) for l, t in path if l < level]
                path.append((level, title))
        else:
            current_body.append(line)

    section = flush()
    if section is not None:
        yield section


def chunk_markdown(
    text: str,
    tokenizer,
    max_tokens: int = 500,
) -> list[dict]:
    chunks: list[dict] = []

    def emit(section: str, title: str, items: list[str]) -> None:
        body = "\n\n".join(items)
        content = f"# {title}\n\n{body}"
        chunks.append({
            "secao": section,
            "content": content,
            "n_tokens": len(
                tokenizer.encode(content, add_special_tokens=False)
            ),
        })

    for section, body in _iter_sections(text):
        if not body:
            continue

        last_title = section.split(" > ")[-1]
        title_tokens = len(
            tokenizer.encode(last_title + "\n\n", add_special_tokens=False)
        )
        budget = max_tokens - title_tokens

        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        current: list[str] = []
        current_tokens = 0

        for para in paragraphs:
            para_tokens = len(
                tokenizer.encode(para, add_special_tokens=False)
            )

            if current and (current_tokens + para_tokens > budget):
                emit(section, last_title, current)
                current = []
                current_tokens = 0

            current.append(para)
            current_tokens += para_tokens

        if current:
            emit(section, last_title, current)

    return chunks
