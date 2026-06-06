import sys
from pathlib import Path

root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))

from search_agent.eval.metrics import exact_match, f1_score, citation_hit


def test_exact_match_normalizes_text():
    assert exact_match("The Brookhaven.", "Brookhaven") == 1.0


def test_f1_score_partial_overlap():
    score = f1_score("Lena Moris was born in Brookhaven", "Brookhaven")
    assert 0 < score < 1


def test_citation_hit():
    assert (
        citation_hit(
            answer="Brookhaven [1]",
            supporting_doc_ids=["doc001", "doc002"],
            cited_doc_ids=["doc002"],
        )
        == 1.0
    )
