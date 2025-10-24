from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os 
import sys 
import pathlib
import tempfile

# Add the parent directory (or its parent) to sys.path to resolve 'scoring' import
core_path = pathlib.Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_path))

from scoring import pronunciation_score, pronunciation_score_with_ai

# Importação dos modelos de transcrição e IA
models_path = pathlib.Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_path))

# Modelos disponíveis (alguns comentados por escolha de arquitetura):
from modelos import (
    Whisper,              # Usado: transcrição local
    # Wav2Vec2,           # Testado: bom para português, mas focado em transcrição
    # DeepSpeech,         # Testado: leve mas limitado em idiomas
    # CoquiSTT,           # Testado: open source mas requer muito fine-tuning
    # FasterWhisper,      # Testado: mais rápido mas sem vantagem para nosso caso
    OpenAITranscriber,    # Usado: transcrição via API
    GeminiTranscriber,    # Usado: transcrição via API
    OpenAIChat,           # Usado: avaliação qualitativa com GPT
    GeminiChat            # Usado: avaliação qualitativa com Gemini
)

app = FastAPI(
    title="API de Avaliação de Pronúncia com IA",
    description="Sistema inteligente que usa GPT/Gemini para avaliar pronúncia de forma qualitativa",
    version="2.0.0"
)

# Modelo principal de transcrição (local, grátis, razoavelmente preciso)
# DESABILITADO: Whisper local consome muita RAM
# whisper_model = Whisper(device='cpu')  # Use 'cuda' se tiver GPU
whisper_model = None  # Usar apenas Gemini (sem Whisper local)

# ============================================================================
# NOTA SOBRE ESCOLHA DE MODELOS:
# ============================================================================
# Testamos diversos modelos STT durante o desenvolvimento:
#
# 1. Whisper (OpenAI) - ESCOLHIDO para transcrição local
#    ✅ Multilíngue, boa precisão, uso offline
#    ❌ Mais lento que alternativas especializadas
#
# 2. Wav2Vec2 - Testado mas não implementado como padrão
#    ✅ Excelente para português brasileiro
#    ❌ Requer fine-tuning por idioma, complexidade adicional
#
# 3. FasterWhisper - Testado mas não necessário
#    ✅ 4x mais rápido que Whisper
#    ❌ Sem ganho significativo para nosso caso de uso
#
# 4. DeepSpeech / Coqui STT - Testados mas descartados
#    ✅ Leves e rápidos
#    ❌ Modelos pré-treinados limitados, requerem treino personalizado
#
# Para AVALIAÇÃO, optamos por LLMs (GPT/Gemini) ao invés de:
# - Algoritmos de similaridade fonética (limitados)
# - Análise de MFCCs e features acústicas (complexo)
# - Modelos especializados em pronúncia (poucos disponíveis)
#
# Razão: LLMs oferecem feedback pedagógico superior
# ============================================================================

async def _salvar_upload_temporario(arquivo: UploadFile) -> str:
    data = await arquivo.read()
    suffix = os.path.splitext(arquivo.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return tmp.name

def _normalizar_provedor(provedor: str) -> str:
    return (provedor or "gemini").lower()  # MUDADO: gemini como padrão

def _transcrever_arquivo(caminho_tmp: str, provedor: str) -> str:
    prov = _normalizar_provedor(provedor)
    if prov == "openai":
        return OpenAITranscriber().transcribe(caminho_tmp)
    if prov == "gemini":
        return GeminiTranscriber().transcribe(caminho_tmp)
    # Whisper local desabilitado (falta de RAM)
    # return whisper_model.transcribe(caminho_tmp)
    # Se pedir whisper, usar gemini
    return GeminiTranscriber().transcribe(caminho_tmp)

async def _transcrever_upload(audio: UploadFile, provedor: str) -> str:
    tmp_path = await _salvar_upload_temporario(audio)
    try:
        return _transcrever_arquivo(tmp_path, provedor)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass

def _resposta_chat_texto(texto: str, provedor: str, sistema: str) -> str:
    prov = _normalizar_provedor(provedor)
    if prov == "gemini":
        return GeminiChat().reply_from_text(texto, system=sistema)
    if prov == "openai":
        return OpenAIChat().reply_from_text(texto, system=sistema)
    raise RuntimeError("Provider sem chat: use 'openai' ou 'gemini'.")

@app.post("/avaliar")
async def avaliar(
    user_id: str = Form(...),
    target_word: str = Form(...),
    audio: UploadFile = Form(...),
    provider: str = Form("whisper"),  # whisper | openai | gemini - para transcrição
    ai_scoring: bool = Form(True),  # Usar IA para avaliação? (padrão: True)
    scoring_provider: str = Form("openai"),  # openai | gemini - para avaliação
    language: str = Form("português"),  # Idioma para contextualizar avaliação
):
    """
    🎯 Endpoint PRINCIPAL para avaliar a pronúncia com IA.
    
    **Fluxo:**
    1. Transcreve o áudio (Whisper local OU OpenAI/Gemini)
    2. Avalia com GPT/Gemini (feedback qualitativo detalhado)
    
    **Parâmetros:**
    - user_id: ID do usuário
    - target_word: Palavra/frase que deveria ser falada
    - audio: Arquivo de áudio (.wav, .mp3, .opus, etc)
    - provider: Modelo para transcrição (whisper=local, openai, gemini)
    - ai_scoring: Se True, usa IA para avaliar (recomendado!)
    - scoring_provider: Qual IA usar na avaliação (openai ou gemini)
    - language: Idioma para contextualizar feedback
    
    **Retorno:**
    - score: Nota de 0 a 100
    - feedback: Análise detalhada da pronúncia
    - suggestions: Dicas para melhorar
    - errors: Lista de erros específicos
    - highlights: O que acertou/errou
    """
    try:
        transcription = await _transcrever_upload(audio, provider)
    except Exception as e:
        return JSONResponse({"error": f"Falha na transcrição ({provider}): {e}"}, status_code=400)

    # Avaliar com IA ou método tradicional
    if ai_scoring:
        score_result = pronunciation_score_with_ai(
            target_word, 
            transcription, 
            provider=scoring_provider,
            language=language
        )
    else:
        score_result = pronunciation_score(target_word, transcription)
    
    # Adicionar metadados
    score_result["user_id"] = user_id
    score_result["transcription_provider"] = provider
    
    return JSONResponse(score_result)

@app.post("/falar")
async def falar(
    audio: UploadFile = Form(...),
    provider: str = Form("openai"),  # openai | gemini
    system: str = Form("Você é um assistente útil que responde de forma curta."),
):
    """
    Fala com o modelo via áudio: transcreve e envia ao LLM selecionado.
    Retorna { transcript, reply }.
    """
    try:
        transcript = await _transcrever_upload(audio, provider)
    except Exception as e:
        return JSONResponse({"error": f"Falha na transcrição ({provider}): {e}"}, status_code=400)

    try:
        reply = _resposta_chat_texto(transcript, provider, system)
    except Exception as e:
        return JSONResponse({"error": f"Falha ao conversar com {provider}: {e}"}, status_code=400)

    return JSONResponse({"transcript": transcript, "reply": reply})

@app.post("/transcrever")
async def transcrever(
    audio: UploadFile = Form(...),
    provider: str = Form("whisper"),  # whisper | openai | gemini
):
    """
    Teste simples: retorna apenas a transcrição do áudio.
    """
    try:
        transcript = await _transcrever_upload(audio, provider)
        return JSONResponse({"transcript": transcript})
    except Exception as e:
        return JSONResponse({"error": f"Falha na transcrição ({provider}): {e}"}, status_code=400)

@app.post("/chat_texto")
async def chat_texto(
    message: str = Form(...),
    provider: str = Form("openai"),  # openai | gemini
    system: str = Form("Você é um assistente útil que responde de forma curta."),
):
    """
    Teste simples: conversa via texto com o LLM (sem áudio).
    """
    try:
        reply = _resposta_chat_texto(message, provider, system)
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"error": f"Falha ao conversar com {provider}: {e}"}, status_code=400)

@app.post("/tutor_pronuncia")
async def tutor_pronuncia(
    message: str = Form(...),
    provider: str = Form("openai"),  # openai | gemini
):
    """
    🎓 NOVO: Tutor de pronúncia interativo via texto.
    
    Conversa natural sobre pronúncia, dúvidas, dicas, exercícios.
    Exemplo: "Como pronunciar 'through'?" ou "Tenho dificuldade com R em inglês"
    """
    system_prompt = """Você é um professor de pronúncia especializado e paciente.

SEU PAPEL:
- Ajudar alunos a melhorar pronúncia em qualquer idioma
- Explicar sons difíceis de forma clara e prática
- Dar exercícios e dicas personalizadas
- Ser encorajador e motivador

ESTILO:
- Use emojis para tornar mais amigável 🎯
- Dê exemplos práticos e comparações
- Se o aluno perguntar sobre uma palavra específica, explique cada som
- Sugira exercícios quando apropriado

FORMATO:
- Seja conciso mas completo
- Use bullets quando listar dicas
- Destaque sons problemáticos com **negrito**"""

    try:
        reply = _resposta_chat_texto(message, provider, system_prompt)
        return JSONResponse({
            "reply": reply,
            "provider": provider,
            "mode": "tutor"
        })
    except Exception as e:
        return JSONResponse({"error": f"Falha ao conversar com tutor: {e}"}, status_code=400)

@app.get("/")
async def root():
    """Página inicial da API com documentação"""
    return {
        "message": "🎯 API de Avaliação de Pronúncia com IA",
        "version": "2.0.0",
        "endpoints": {
            "/avaliar": "Avaliar pronúncia com feedback de IA (POST)",
            "/falar": "Conversar via áudio com IA (POST)",
            "/transcrever": "Apenas transcrever áudio (POST)",
            "/chat_texto": "Chat via texto (POST)",
            "/tutor_pronuncia": "Tutor interativo de pronúncia (POST)",
            "/docs": "Documentação interativa Swagger"
        },
        "providers": {
            "transcription": ["whisper", "openai", "gemini"],
            "scoring": ["openai", "gemini"],
            "chat": ["openai", "gemini"]
        },
        "features": [
            "✅ Transcrição de áudio com múltiplos modelos",
            "✅ Avaliação qualitativa com GPT/Gemini",
            "✅ Feedback detalhado e personalizado",
            "✅ Sugestões de melhoria",
            "✅ Tutor de pronúncia interativo",
            "✅ Chat por áudio ou texto"
        ]
    }