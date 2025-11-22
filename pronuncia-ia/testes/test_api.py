"""Testes de integração da API de Avaliação de Pronúncia (Pytest).

Requer servidor FastAPI rodando em http://localhost:8000.
Cada teste realiza chamada real; se o servidor não estiver ativo os testes serão marcados como pendentes (xfail) ou falharão.
"""
import os
import pytest
import requests

BASE_URL = os.getenv("PRONUNCIACORE_SERVER", "http://localhost:8000")


def _server_up() -> bool:
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not _server_up(), reason="Servidor FastAPI não está acessível em BASE_URL")
def test_health_check():
    resp = requests.get(f"{BASE_URL}/", timeout=10)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)


@pytest.mark.skipif(not _server_up(), reason="Servidor FastAPI não está acessível em BASE_URL")
def test_avaliar_pronuncia_perfeita():
    payload = {
        "expected": "Olá, como você está hoje?",
        "predicted": "Olá, como você está hoje?",
        "ai_scoring": True,
        "scoring_provider": "gemini",
        "language": "pt-BR",
    }
    resp = requests.post(f"{BASE_URL}/avaliar", json=payload, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("score") is not None
    assert data.get("match") in (True, False)
    assert data.get("method")
    assert "feedback" in data


@pytest.mark.skipif(not _server_up(), reason="Servidor FastAPI não está acessível em BASE_URL")
def test_avaliar_pronuncia_com_erros():
    payload = {
        "expected": "O clima está maravilhoso hoje",
        "predicted": "O clima esta maraviioso hoje",
        "ai_scoring": True,
        "scoring_provider": "gemini",
        "language": "pt-BR",
    }
    resp = requests.post(f"{BASE_URL}/avaliar", json=payload, timeout=30)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("score"), (int, float))
    # Deve fornecer feedback mais rico quando há erros
    assert "feedback" in data
    # Erros opcionais; se presentes devem ser lista
    if data.get("errors"):
        assert isinstance(data["errors"], list)


@pytest.mark.skipif(not _server_up(), reason="Servidor FastAPI não está acessível em BASE_URL")
def test_metodo_tradicional_sem_ia():
    payload = {
        "expected": "Olá mundo",
        "predicted": "Olá mundo",
        "ai_scoring": False,
    }
    resp = requests.post(f"{BASE_URL}/avaliar", json=payload, timeout=15)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("method")
    assert data.get("score") is not None
    assert "feedback" in data
