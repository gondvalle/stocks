import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from core.utils import safe_div
from core.metrics import compute_p_fcf


def test_safe_div():
    assert safe_div(10, 2) == 5
    assert safe_div(1, 0) != safe_div(1, 0)


def test_compute_p_fcf():
    income = {"Total Revenue": [100], "Net Income": [10]}
    cash = {"Operating Cash Flow": [50], "Capital Expenditure": [-20]}
    assert round(compute_p_fcf(100, income, cash), 2) == 3.33
