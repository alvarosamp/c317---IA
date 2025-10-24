import os
import mimetypes
from typing import Optional

# Imports de bibliotecas STT (opcionais - podem não estar instaladas)
try:
    from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
except Exception:
    pipeline = None
    Wav2Vec2ForCTC = None
    Wav2Vec2Processor = None

try:
    import deepspeech
except Exception:
    deepspeech = None

try:
    import torch
except Exception:
    torch = None

try:
    import librosa
except Exception:
    librosa = None

try:
    import numpy as np
except Exception:
    np = None

try:
    import wave
except Exception:
    wave = None

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

try:
    import coqui
except Exception:
    coqui = None

# SDKs opcionais (não falhar no import do módulo inteiro se não instalados)
try:
    from openai import OpenAI  # pip install openai>=1.0
except Exception:
    OpenAI = None

try:
    import google.generativeai as genai  # pip install google-generativeai
except Exception:
    genai = None

# Classe para o Whisper
class Whisper:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = pipeline('automatic-speech-recognition', model='openai/whisper-small', device=self.device)

    def transcribe(self, audio_path):
        """
        Transcreve o áudio usando o modelo Whisper.
        """
        result = self.model(audio_path)
        return result['text']

# Classe para o Wav2Vec2
class Wav2Vec2:
    def __init__(self, model_name='jonatasgrosman/wav2vec2-large-xlsr-53-portuguese', device='cuda'):
        """
        Inicializa o modelo Wav2Vec2 com o nome e o dispositivo especificado.
        """
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)

    def transcribe(self, audio_path):
        """
        Transcreve o áudio usando o modelo Wav2Vec2.
        """
        # Carregar e pré-processar o áudio
        audio_input, _ = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio_input, return_tensors='pt', sampling_rate=16000)

        # Inferir com o modelo
        with torch.no_grad():
            logits = self.model(input_values=inputs.input_values).logits

        # Decodificar a transcrição
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription

# Classe para o DeepSpeech
class DeepSpeech:
    def __init__(self, model_path):
        """
        Inicializa o modelo DeepSpeech a partir do caminho do arquivo do modelo.
        """
        self.model = deepspeech.Model(model_path)

    def transcribe(self, audio_path):
        """
        Transcreve o áudio usando o modelo DeepSpeech.
        """
        with wave.open(audio_path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            data16 = np.frombuffer(frames, dtype=np.int16)
        transcription = self.model.stt(data16)
        return transcription

# Classe para o Coqui STT
class CoquiSTT:
    def __init__(self, model_path):
        """
        Inicializa o modelo Coqui STT a partir do caminho do arquivo do modelo.
        """
        self.model = coqui.stt.Model(model_path)

    def transcribe(self, audio_path):
        """
        Transcreve o áudio usando o modelo Coqui STT.
        """
        with wave.open(audio_path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            data16 = np.frombuffer(frames, dtype=np.int16)
        transcription = self.model.stt(data16)
        return transcription

# Classe para o Faster Whisper
class FasterWhisper:
    def __init__(self, model_size='small', device='cuda'):
        """
        Inicializa o modelo Faster Whisper com o tamanho e dispositivo especificado.
        """
        self.device = device
        self.model = WhisperModel(model_size, device=self.device)
    def transcribe(self, audio_path):
        """
        Transcreve o áudio usando o modelo Faster Whisper.
        """
        segments, info = self.model.transcribe(audio_path)
        transcription = " ".join([segment.text for segment in segments])
        return transcription
class OpenAITranscriber:
    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("Pacote 'openai' não instalado. Use: pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY não configurada no ambiente.")
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe")  # alternativo: 'whisper-1'

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            resp = self.client.audio.transcriptions.create(model=self.model, file=f)
        return getattr(resp, "text", "")
class GeminiTranscriber:
    def __init__(self, model: Optional[str] = None):
        if genai is None:
            raise RuntimeError("Pacote 'google-generativeai' não instalado. Use: pip install google-generativeai")
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY/GEMINI_API_KEY não configurada no ambiente.")
        genai.configure(api_key=api_key)
        self.model_name = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = genai.GenerativeModel(self.model_name)

    def transcribe(self, audio_path: str) -> str:
        with open(audio_path, "rb") as f:
            data = f.read()
        mime = mimetypes.guess_type(audio_path)[0] or "audio/wav"
        prompt = "Transcreva o áudio exatamente como falado, mantendo o idioma."
        resp = self.model.generate_content([prompt, {"mime_type": mime, "data": data}])
        return getattr(resp, "text", "")


class OpenAIChat:
    def __init__(self, model: Optional[str] = None):
        if OpenAI is None:
            raise RuntimeError("Pacote 'openai' não instalado. Use: pip install openai")
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY não configurada no ambiente.")
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def reply(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content

    def reply_from_text(self, user_text: str, system: str = "Você é um assistente útil."):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        return self.reply(messages)

class GeminiChat:
    def __init__(self, model: Optional[str] = None):
        print(f"[DEBUG] 🔧 Inicializando GeminiChat...")
        
        if genai is None:
            print(f"[DEBUG] ❌ Pacote google-generativeai não está instalado!")
            raise RuntimeError("Pacote 'google-generativeai' não instalado. Use: pip install google-generativeai")
        
        print(f"[DEBUG] ✅ Pacote google-generativeai disponível")
        
        # Verificar variáveis de ambiente
        google_key = os.getenv("GOOGLE_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")
        
        print(f"[DEBUG] GOOGLE_API_KEY encontrada: {google_key is not None}")
        print(f"[DEBUG] GEMINI_API_KEY encontrada: {gemini_key is not None}")
        
        api_key = google_key or gemini_key
        if not api_key:
            print(f"[DEBUG] ❌ Nenhuma chave de API encontrada!")
            raise RuntimeError("GOOGLE_API_KEY/GEMINI_API_KEY não configurada no ambiente.")
        
        print(f"[DEBUG] ✅ Chave de API encontrada: {api_key[:20]}...")
        
        genai.configure(api_key=api_key)
        self.model_name = model or os.getenv("GEMINI_CHAT_MODEL", os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        print(f"[DEBUG] 📦 Usando modelo: {self.model_name}")
        
        self.model = genai.GenerativeModel(self.model_name)
        print(f"[DEBUG] ✅ GeminiChat inicializado com sucesso!")

    def reply(self, messages: list[dict]) -> str:
        print(f"[DEBUG] 💬 GeminiChat.reply() chamado com {len(messages)} mensagens")
        # Concatena system + turns simples em texto
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
        prompt = (sys_msg + "\n\n" if sys_msg else "") + "\n\n".join(user_msgs)
        print(f"[DEBUG] 📤 Enviando prompt para Gemini (tamanho: {len(prompt)} chars)...")
        resp = self.model.generate_content(prompt)
        result = getattr(resp, "text", "")
        print(f"[DEBUG] 📥 Resposta recebida (tamanho: {len(result)} chars)")
        return result

    def reply_from_text(self, user_text: str, system: str = "Você é um assistente útil."):
        print(f"[DEBUG] 💬 GeminiChat.reply_from_text() chamado")
        return self.reply([{"role": "system", "content": system}, {"role": "user", "content": user_text}])