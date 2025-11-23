
import os
from pathlib import Path
import pytest
import requests

AUDIO_PATH = Path(os.getenv("PRONUNCIACORE_AUDIO", "/Users/alvarosamp/Documents/Projetos/p8/Top1/c317---IA/pronuncia-ia/audioteste/audio.opus"))
SERVER_URL = os.environ.get("PRONUNCIACORE_SERVER", "http://127.0.0.1:8000")
BASE_URL = SERVER_URL.rstrip("/")


def _up() -> bool:
    try:
        return requests.get(f"{BASE_URL}/", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = [pytest.mark.integration]


@pytest.mark.skipif(not AUDIO_PATH.exists(), reason="Arquivo de áudio ausente")
@pytest.mark.skipif(not _up(), reason="Servidor indisponível")
def test_upload_audio_mock_provider():
    url = f"{BASE_URL}/avaliar"
    with AUDIO_PATH.open("rb") as fh:
        files = {"audio": (AUDIO_PATH.name, fh, "audio/opus")}
        data = {
            "user_id": "test_user",
            "action": "evaluate",
            "target_word": "O rato roeu a roupa do rei de Roma",
            "ai_scoring": "false",
            "provider": "mock",
        }
        resp = requests.post(url, files=files, data=data, timeout=60)
    assert resp.status_code == 200
    body = resp.json()
    assert "score" in body
    assert body.get("method") in {"levenshtein", "ai-gemini", "ai-openai"}
