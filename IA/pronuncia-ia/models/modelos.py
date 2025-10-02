from transformers import pipeline, Wav2Vec2ForCTC, Wav2Vec2Processor
import deepspeech
import torch
import librosa
import numpy as np
import wave
from faster_whisper import WhisperModel
import coqui

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