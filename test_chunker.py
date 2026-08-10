"""Tests for chunk_markdown."""

import chunker


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


def test_multiple_sections():
    text = (
        "# Currículo\n"
        "## Artigos Científicos\n"
        "Parágrafo um sobre publicação.\n"
        "\n"
        "Parágrafo dois sobre publicação.\n"
        "\n"
        "## Orientações Acadêmicas Concluídas\n"
        "Orientação em nível de mestrado.\n"
    )
    chunks = chunker.chunk_markdown(text, FakeTokenizer(), max_tokens=100)
    sections = [c["secao"] for c in chunks]
    assert "Currículo > Artigos Científicos" in sections
    assert "Currículo > Orientações Acadêmicas Concluídas" in sections
    assert all(c["content"] for c in chunks)
    assert all(c["n_tokens"] > 0 for c in chunks)


def test_single_paragraph_over_max_tokens():
    text = "# Seção\n"
    big_para = "palavra " * 200
    chunks = chunker.chunk_markdown(text + big_para, FakeTokenizer(), max_tokens=10)
    assert len(chunks) == 1
    assert chunks[0]["secao"] == "Seção"
    assert chunks[0]["content"].endswith(big_para.strip())
    assert chunks[0]["content"].startswith("# Seção")
    assert chunks[0]["n_tokens"] > 200


def test_empty_section():
    text = "# Seção A\nConteúdo.\n\n## Seção Vazia\n\n# Seção B\nOutro.\n"
    chunks = chunker.chunk_markdown(text, FakeTokenizer(), max_tokens=100)
    sections = [c["secao"] for c in chunks]
    assert "Seção Vazia" not in sections
    assert "Seção A" in sections
    assert "Seção B" in sections
