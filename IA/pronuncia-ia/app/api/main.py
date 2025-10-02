from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os 
import sys 
import pathlib

# Add the parent directory (or its parent) to sys.path to resolve 'scoring' import
scoring_path = pathlib.Path(__file__).parent / "scoring.py"
if not scoring_path.exists():
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
else:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

from scoring import pronunciation_score

#Testando com o whisper
models_path = pathlib.Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_path))
from models.modelos import Whisper, Wav2Vec2, DeepSpeech, CoquiSTT, FasterWhisper
app = FastAPI()
whisper_model = Whisper()

@app.post("/avaliar")
async def avaliar(user_id: str = Form(...), target_word: str = Form(...), audio: UploadFile = Form(...)):
    """
    Endpoint para avaliar a pronúncia de uma palavra pelo usuário.
    """
    # Processar o áudio e transcrição
    transcription = whisper_model.transcribe(audio)  # Usando o Whisper para transcrição

    # Calcular a pontuação de pronúncia
    score = pronunciation_score(target_word, transcription)

    return JSONResponse(score)