"""Teste de avaliação de pronúncia com e sem IA (versão Pytest).

Inclui:
 - Casos parametrizados para método tradicional
 - Teste com IA (skip se sem credenciais)
 - Comparação entre métodos
"""
import os
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "core"))

from scoring import pronunciation_score, pronunciation_score_with_ai  # type: ignore

pytestmark = [pytest.mark.integration]


@pytest.mark.parametrize(
    "esperado,falado",
    [
        ("olá", "olá"),
        ("olá", "ola"),
        ("pronúncia", "pronuncia"),
        ("clima maravilhoso", "clima maraviioso"),
    ],
)
def test_metodo_tradicional_varios_casos(esperado, falado):
    resultado = pronunciation_score(esperado, falado)
    assert "score" in resultado
    assert resultado["method"] == "levenshtein"
    assert isinstance(resultado["hit"], bool)


def _provider_disponivel() -> str | None:
    gemini_ok = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    if gemini_ok:
        return "gemini"
    if openai_ok:
        return "openai"
    return None


@pytest.mark.skipif(_provider_disponivel() is None, reason="Nenhum provider IA configurado")
def test_avaliacao_com_ia_basica():
    provider = _provider_disponivel()
    resultado = pronunciation_score_with_ai("olá", "ola", provider=provider, language="pt-BR")
    assert "score" in resultado
    assert resultado["method"].startswith("ai-")
    assert "feedback" in resultado


@pytest.mark.skipif(_provider_disponivel() is None, reason="Nenhum provider IA configurado")
def test_comparacao_tradicional_vs_ia():
    esperado, falado = "pronúncia", "pronuncia"
    trad = pronunciation_score(esperado, falado)
    ia = pronunciation_score_with_ai(esperado, falado, provider=_provider_disponivel())
    assert trad["score"] >= 0
    assert ia["score"] >= 0
    # IA deve retornar feedback possivelmente mais rico
    assert len(ia.get("feedback", "")) >= len(trad.get("feedback", ""))
