import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from core.scoring import evaluate_company, THRESHOLDS


def test_hard_gate_reduces_score():
    row = {
        "target_vs_price": 0.5,
        "pe_vs_sector": 0.5,
        "ev_ebitda_vs_sector": 0.5,
        "p_fcf_vs_sector": 0.5,
    }
    score, checks = evaluate_company(row, {})
    assert score == 0
    row["target_vs_price"] = 2
    score2, _ = evaluate_company(row, {})
    assert score2 > 0
