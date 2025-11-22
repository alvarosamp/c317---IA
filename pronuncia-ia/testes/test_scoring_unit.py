"""Testes unitários do módulo de scoring (Levenshtein)."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "core"))
from scoring import pronunciation_score, string_similarity  # type: ignore

pytestmark = [pytest.mark.unit]


def test_string_similarity_perfeito():
    assert string_similarity("Olá", "Olá") == pytest.approx(1.0)


def test_string_similarity_vazio():
    assert string_similarity("", "") == pytest.approx(1.0)


def test_string_similarity_diferenca():
    val = string_similarity("banana", "ban")
    assert 0 < val < 1


def test_pronunciation_score_hit():
    r = pronunciation_score("Teste", "Teste")
    assert r["hit"] is True
    assert r["score"] == pytest.approx(100.0)


def test_pronunciation_score_parcial():
    r = pronunciation_score("pronúncia", "pronuncia")
    assert r["hit"] is False
    assert r["score"] < 100
    assert r["method"] == "levenshtein"
