"""
Teste rápido para enviar um áudio ao endpoint /avaliar
Como usar:
  source /Users/alvarosamp/Documents/Projetos/p8/Top1/.venv/bin/activate
  pip install requests
  python tests/test_evaluate_audio.py

O script assume que o servidor está rodando em http://127.0.0.1:8000
Altere SERVER_URL se necessário.
"""
import os
import requests
from pathlib import Path

# Caminho do áudio fornecido
AUDIO_PATH = Path("/Users/alvarosamp/Documents/Projetos/p8/Top1/c317---IA/pronuncia-ia/audioteste/audio.opus")
SERVER_URL = os.environ.get("PRONUNCIACORE_SERVER", "http://127.0.0.1:8000")

if not AUDIO_PATH.exists():
    print(f"Arquivo de áudio não encontrado: {AUDIO_PATH}")
    raise SystemExit(1)

url = f"{SERVER_URL}/avaliar"
# usamos action=evaluate; definimos ai_scoring=false para evitar chamadas a LLM se você não tiver credenciais
files = {"audio": (AUDIO_PATH.name, open(AUDIO_PATH, "rb"), "audio/opus")}
data = {
    "user_id": "test_user",
    "action": "evaluate",
    "target_word": "O rato roeu a roupa do rei de Roma",
    "ai_scoring": "false",
    "provider": "mock",
}

print(f"Enviando POST {url} com arquivo {AUDIO_PATH}...")
resp = requests.post(url, files=files, data=data, timeout=60)

print("Status:", resp.status_code)
try:
    print("JSON response:")
    print(resp.json())
except Exception:
    print("Response text:")
    print(resp.text)
finally:
    files['audio'][1].close()
