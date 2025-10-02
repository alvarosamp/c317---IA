# Criando testes com pytest e mock para os modelos de transcrição
import pytest
from unittest import mock
import sys
import pathlib

# Mock all external dependencies before any imports
mock_modules = {
    'transformers': mock.MagicMock(),
    'deepspeech': mock.MagicMock(),
    'torch': mock.MagicMock(),
    'librosa': mock.MagicMock(),
    'numpy': mock.MagicMock(),
    'wave': mock.MagicMock(),
    'faster_whisper': mock.MagicMock(),
    'coqui': mock.MagicMock(),
}

# Apply mocks to sys.modules
for module_name, mock_module in mock_modules.items():
    sys.modules[module_name] = mock_module

# Configure specific mock behaviors
mock_modules['transformers'].pipeline = mock.MagicMock()
mock_modules['transformers'].Wav2Vec2Processor = mock.MagicMock()
mock_modules['transformers'].Wav2Vec2ForCTC = mock.MagicMock()
mock_modules['deepspeech'].Model = mock.MagicMock()
mock_modules['coqui'].stt = mock.MagicMock()
mock_modules['faster_whisper'].WhisperModel = mock.MagicMock()

# Add the models directory to the Python path
models_path = pathlib.Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_path))

# Now import the classes
from modelos import Whisper, Wav2Vec2, DeepSpeech, CoquiSTT, FasterWhisper

# Tests for the speech-to-text models


# Teste para o modelo Whisper  
def test_whisper():
    # Configure the global mock before creating the model
    mock_model_instance = mock.MagicMock()
    mock_model_instance.return_value = {'text': 'test transcription'}
    mock_modules['transformers'].pipeline.return_value = mock_model_instance
    
    whisper_model = Whisper(device='cpu')
    transcription = whisper_model.transcribe("fake_audio.wav")
    
    assert transcription == 'test transcription'
    mock_modules['transformers'].pipeline.assert_called_once_with('automatic-speech-recognition', 
                                                                 model='openai/whisper-small', 
                                                                 device='cpu')


# Teste para o modelo Wav2Vec2
def test_wav2vec2():
    # Configure processor mock
    mock_processor_instance = mock.MagicMock()
    mock_modules['transformers'].Wav2Vec2Processor.from_pretrained.return_value = mock_processor_instance
    
    # Configure model mock
    mock_model_instance = mock.MagicMock()
    mock_model_instance.to.return_value = mock_model_instance  # for .to(device)
    mock_modules['transformers'].Wav2Vec2ForCTC.from_pretrained.return_value = mock_model_instance
    
    # Configure librosa mock
    mock_modules['librosa'].load.return_value = (mock.MagicMock(), 16000)
    
    # Configure processor input handling
    mock_inputs = mock.MagicMock()
    mock_inputs.input_values = mock.MagicMock()
    mock_processor_instance.return_value = mock_inputs
    
    # Configure model output
    mock_logits = mock.MagicMock()
    mock_model_return = mock.MagicMock()
    mock_model_return.logits = mock_logits 
    mock_model_instance.return_value = mock_model_return
    
    # Configure torch operations
    mock_context_manager = mock.MagicMock()
    mock_context_manager.__enter__ = mock.MagicMock()
    mock_context_manager.__exit__ = mock.MagicMock()
    mock_modules['torch'].no_grad.return_value = mock_context_manager
    
    predicted_ids = mock.MagicMock()
    mock_modules['torch'].argmax.return_value = predicted_ids
    
    # Configure processor decode
    mock_processor_instance.decode.return_value = "test transcription"
    
    wav2vec_model = Wav2Vec2(device='cpu')
    transcription = wav2vec_model.transcribe("fake_audio.wav")
    
    assert transcription == "test transcription"


# Teste para o modelo DeepSpeech
def test_deepspeech():
    # Configure DeepSpeech model mock
    mock_model_instance = mock.MagicMock()
    mock_model_instance.stt.return_value = "deepspeech transcription"
    mock_modules['deepspeech'].Model.return_value = mock_model_instance
    
    # Configure wave mock
    mock_wave_file = mock.MagicMock()
    mock_wave_file.__enter__ = mock.MagicMock(return_value=mock_wave_file)
    mock_wave_file.__exit__ = mock.MagicMock(return_value=None)
    mock_wave_file.readframes.return_value = b'fake_audio_data'
    mock_wave_file.getnframes.return_value = 1000
    mock_modules['wave'].open.return_value = mock_wave_file
    
    # Configure numpy mock
    mock_data16 = mock.MagicMock()
    mock_modules['numpy'].frombuffer.return_value = mock_data16
    
    deepspeech_model = DeepSpeech(model_path="path_to_model.pbmm")
    transcription = deepspeech_model.transcribe("fake_audio.wav")
    
    assert transcription == "deepspeech transcription"
    mock_modules['deepspeech'].Model.assert_called_once_with("path_to_model.pbmm")


# Teste para o modelo Coqui STT
def test_coqui():
    # Configure Coqui STT model mock
    mock_model_instance = mock.MagicMock()
    mock_model_instance.stt.return_value = "coqui transcription"
    mock_modules['coqui'].stt.Model.return_value = mock_model_instance
    
    # Configure wave mock (same as DeepSpeech)
    mock_wave_file = mock.MagicMock()
    mock_wave_file.__enter__ = mock.MagicMock(return_value=mock_wave_file)
    mock_wave_file.__exit__ = mock.MagicMock(return_value=None)
    mock_wave_file.readframes.return_value = b'fake_audio_data'
    mock_wave_file.getnframes.return_value = 1000
    mock_modules['wave'].open.return_value = mock_wave_file
    
    # Configure numpy mock
    mock_data16 = mock.MagicMock()
    mock_modules['numpy'].frombuffer.return_value = mock_data16
    
    coqui_model = CoquiSTT(model_path="path_to_model.tflite")
    transcription = coqui_model.transcribe("fake_audio.wav")
    
    assert transcription == "coqui transcription"
    mock_modules['coqui'].stt.Model.assert_called_once_with("path_to_model.tflite")


# Teste para o modelo Faster Whisper
def test_faster_whisper():
    # Configure Faster Whisper model mock
    mock_model_instance = mock.MagicMock()
    
    # Configure segments mock - FasterWhisper returns segments with .text attribute
    mock_segment1 = mock.MagicMock()
    mock_segment1.text = "faster"
    mock_segment2 = mock.MagicMock()
    mock_segment2.text = "whisper"
    mock_segment3 = mock.MagicMock()
    mock_segment3.text = "transcription"
    
    mock_segments = [mock_segment1, mock_segment2, mock_segment3]
    mock_info = mock.MagicMock()
    
    mock_model_instance.transcribe.return_value = (mock_segments, mock_info)
    mock_modules['faster_whisper'].WhisperModel.return_value = mock_model_instance
    
    faster_whisper_model = FasterWhisper(model_size='small', device='cpu')
    transcription = faster_whisper_model.transcribe("fake_audio.wav")
    
    assert transcription == "faster whisper transcription"
    mock_modules['faster_whisper'].WhisperModel.assert_called_once_with('small', device='cpu')
