"""Testes em memória usando TestClient.

Este arquivo substitui o script anterior por testes unitários que executam a
aplicação FastAPI em memória (sem necessidade de uvicorn) e enviam o áudio
local `audioteste/audio.opus` para os endpoints `/transcrever` e `/avaliar`.

Observações:
- Usa o provider `mock` para transcrição, portanto não depende de chaves externas.
- Para avaliação sem IA, usa `ai_scoring=false` e o método tradicional.
"""
from pathlib import Path
from fastapi.testclient import TestClient
import pytest

from app.api.main import app  # importa a app já configurada


ROOT_AUDIO = Path("/Users/alvarosamp/Documents/Projetos/p8/Top1/c317---IA/pronuncia-ia/audioteste/audio.opus")
client = TestClient(app)


def _ensure_audio_exists():
    assert ROOT_AUDIO.exists(), f"Arquivo de áudio não encontrado: {ROOT_AUDIO}"


@pytest.mark.unit
def test_transcrever_endpoint_com_audio_opus_mock():
    """Verifica que o endpoint /transcrever retorna uma transcrição quando passado áudio."""
    _ensure_audio_exists()
    with ROOT_AUDIO.open("rb") as fh:
        files = {"audio": (ROOT_AUDIO.name, fh, "audio/opus")}
        data = {"provider": "mock"}
        r = client.post("/transcrever", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    assert "transcript" in body
    assert isinstance(body["transcript"], str)


@pytest.mark.unit
def test_avaliar_action_transcribe_via_avaliar_mock():
    """Chama /avaliar com action=transcribe e provider=mock para garantir retorno correto."""
    _ensure_audio_exists()
    with ROOT_AUDIO.open("rb") as fh:
        files = {"audio": (ROOT_AUDIO.name, fh, "audio/opus")}
        data = {"provider": "mock", "action": "transcribe"}
        r = client.post("/avaliar", files=files, data=data)
    assert r.status_code == 200
    j = r.json()
    assert "transcription" in j
    assert isinstance(j["transcription"], str)


@pytest.mark.unit
def test_avaliar_evaluate_sem_ia_com_audio_opus():
    """Chama /avaliar com ai_scoring=false (método tradicional) usando provider=mock."""
    _ensure_audio_exists()
    target = "O rato roeu a roupa do rei de Roma"
    with ROOT_AUDIO.open("rb") as fh:
        files = {"audio": (ROOT_AUDIO.name, fh, "audio/opus")}
        data = {
            "provider": "mock",
            "ai_scoring": "false",
            "action": "evaluate",
            "target_word": target,
        }
        r = client.post("/avaliar", files=files, data=data)
    assert r.status_code == 200
    body = r.json()
    # Campos esperados
    assert "score" in body
    assert "transcription" in body
    assert body.get("transcription") == "o rato roeu a roupa do rei de roma"
    assert body.get("user_id") is None or isinstance(body.get("user_id"), (str, type(None)))

