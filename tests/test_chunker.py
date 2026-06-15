"""
Tests for the chunking pipeline.

Senior note: we test the chunker before it touches any real data.
  Chunking bugs are silent — bad chunks produce bad retrieval and
  you'll never know why unless you test the boundaries explicitly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.chunker import DocumentChunker, _split_into_chunks
from src.models import Paper


def make_paper(**kwargs) -> Paper:
    defaults = dict(
        arxiv_id="2401.00001",
        title="Test Paper on RAG Systems",
        abstract="This paper presents a novel approach to retrieval augmented generation. " * 10,
        authors=["Alice Smith", "Bob Jones"],
        categories=["cs.AI"],
        published="2024-01-01",
        pdf_url="https://arxiv.org/pdf/2401.00001",
    )
    return Paper(**{**defaults, **kwargs})


def test_split_produces_chunks():
    text = "The quick brown fox. " * 200
    chunks = _split_into_chunks(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1, "Long text should produce multiple chunks"


def test_split_overlap():
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = _split_into_chunks(text, chunk_size=50, overlap=10)
    # Verify last words of chunk N appear in chunk N+1
    if len(chunks) >= 2:
        last_words_c0 = chunks[0].split()[-5:]
        next_chunk_text = chunks[1]
        overlap_found = any(w in next_chunk_text for w in last_words_c0)
        assert overlap_found, "Overlap words should appear in next chunk"


def test_chunker_produces_abstract_chunk():
    chunker = DocumentChunker()
    paper = make_paper()
    chunks = chunker.chunk_paper(paper)
    sections = [c.source_section for c in chunks]
    assert "abstract" in sections, "Should always produce an abstract chunk"


def test_chunker_ids_are_unique():
    chunker = DocumentChunker()
    paper = make_paper()
    chunks = chunker.chunk_paper(paper)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "All chunk IDs must be unique"


def test_chunker_min_length_filter():
    chunker = DocumentChunker()
    paper = make_paper(abstract="Too short.")
    chunks = chunker.chunk_paper(paper)
    for c in chunks:
        assert len(c.text) >= 50, f"Chunk too short: '{c.text[:50]}'"


def test_chunk_paper_multiple():
    chunker = DocumentChunker()
    papers = [make_paper(arxiv_id=f"240{i}.00001") for i in range(5)]
    all_chunks = chunker.chunk_papers(papers)
    assert len(all_chunks) > 0
    # Verify arxiv_ids are all valid
    paper_ids = {p.arxiv_id for p in papers}
    for c in all_chunks:
        assert c.arxiv_id in paper_ids


if __name__ == "__main__":
    tests = [
        test_split_produces_chunks,
        test_split_overlap,
        test_chunker_produces_abstract_chunk,
        test_chunker_ids_are_unique,
        test_chunker_min_length_filter,
        test_chunk_paper_multiple,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {t.__name__}: {e}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
