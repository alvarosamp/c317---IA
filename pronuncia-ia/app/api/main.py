try:
    from fastapi import FastAPI, UploadFile, Form, Request, File
    from fastapi.responses import JSONResponse
except Exception as e:
    raise RuntimeError(
        "Dependência ausente: instale FastAPI e Uvicorn (por exemplo: `pip install fastapi uvicorn`) antes de executar este módulo."
    ) from e

from app.api.schemas import EvaluateResponse, TranscribeResponse
from pydantic import BaseModel
from typing import Optional
import os 
import sys 
import pathlib
import tempfile
import uuid
import base64
from pathlib import Path

# Carregar variáveis de ambiente o mais cedo possível
try:
    from dotenv import load_dotenv
    # Tenta múltiplos caminhos para .env para suportar execuções diferentes
    possible_envs = [
        Path(__file__).parent.parent.parent / ".env",        # pronuncia-ia/.env
        Path(__file__).resolve().parents[3] / ".env",         # repo root .env (c317---IA/.env)
        Path.cwd() / ".env",                                  # cwd/.env
    ]
    _env_loaded = None
    for p in possible_envs:
        try:
            if p.exists():
                load_dotenv(dotenv_path=p)
                _env_loaded = p
                break
        except Exception:
            continue
    if _env_loaded:
        print(f"[DEBUG] main.py: .env carregado de {_env_loaded}")
    else:
        print(f"[DEBUG] main.py: nenhum .env encontrado em {[str(p) for p in possible_envs]}")
except Exception as _e:
    print(f"[DEBUG] main.py: falha ao carregar .env antecipadamente: {_e}")

# Add the parent directory (or its parent) to sys.path to resolve 'scoring' import
core_path = pathlib.Path(__file__).parent.parent / "core"
sys.path.insert(0, str(core_path))

from app.core.scoring import pronunciation_score, pronunciation_score_with_ai

# Importação dos modelos de transcrição e IA
models_path = pathlib.Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_path))

# Modelos disponíveis (alguns comentados por escolha de arquitetura):
try:
    from app.models.modelos import (
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
except Exception:
    # Durante desenvolvimento local o pacote pode não estar resolvido via package imports.
    # Fallback para import via path inserido anteriormente.
    # If fallback fails, raise so the error surfaces during startup rather than silently failing later
    from modelos import (
        Whisper,
        OpenAITranscriber,
        GeminiTranscriber,
        OpenAIChat,
        GeminiChat,
    )

app = FastAPI(
    title="API de Avaliação de Pronúncia com IA",
    description="Sistema inteligente que usa GPT/Gemini para avaliar pronúncia de forma qualitativa",
    version="2.0.0"
)


@app.get("/debug_env")
async def debug_env():
    """Endpoint temporário para verificar se as chaves de API estão visíveis no processo.

    Retorna booleans e um valor mascarado (primeiros 6 chars) para ajudar debug sem expor a chave inteira.
    """
    def mask(v: str | None) -> str | None:
        if not v:
            return None
        return v[:6] + "..." if len(v) > 6 else "***"

    gem = os.getenv("GEMINI_API_KEY")
    goo = os.getenv("GOOGLE_API_KEY")
    return JSONResponse({
        "GEMINI_present": bool(gem),
        "GOOGLE_present": bool(goo),
        "GEMINI_masked": mask(gem),
        "GOOGLE_masked": mask(goo),
        "python_executable": sys.executable,
    })
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


# Função dummy para simular caminho de áudio (já que não há upload de arquivo)
def _get_audio_path(audio_dict):
    # Aqui você pode implementar lógica para buscar o arquivo pelo nome, se necessário
    # Por enquanto, só retorna o nome
    return audio_dict.get("name", "")

def _normalizar_provedor(provedor: str) -> str:
    return (provedor or "gemini").lower()  # MUDADO: gemini como padrão

def _transcrever_arquivo(caminho_tmp: str, provedor: str) -> str:
    prov = _normalizar_provedor(provedor)
    # Mock provider for local testing without API keys
    if prov == "mock":
        # Return a stable expected transcription for tests
        return "o rato roeu a roupa do rei de roma"
    if prov == "openai":
        return OpenAITranscriber().transcribe(caminho_tmp)
    if prov == "gemini":
        return GeminiTranscriber().transcribe(caminho_tmp)
    # Whisper local desabilitado (falta de RAM)
    # return whisper_model.transcribe(caminho_tmp)
    # Se pedir whisper, usar gemini
    return GeminiTranscriber().transcribe(caminho_tmp)


# Função para processar upload de arquivo e transcrever
async def _transcrever_upload(audio: UploadFile, provedor: str) -> str:
    data = await audio.read()
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return _transcrever_arquivo(tmp_path, provedor)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


# _transcrever_upload não é mais usado pelo endpoint /avaliar (JSON)


def _resposta_chat_texto(texto: str, provedor: str, sistema: str) -> str:
    prov = _normalizar_provedor(provedor)
    if prov == "gemini":
        return GeminiChat().reply_from_text(texto, system=sistema)
    if prov == "openai":
        return OpenAIChat().reply_from_text(texto, system=sistema)
    raise RuntimeError("Provider sem chat: use 'openai' ou 'gemini'.")
@app.post("/avaliar")
async def avaliar(
    request: Request,
    user_id: Optional[str] = Form(None),
    action: str = Form("evaluate"),  # transcribe | evaluate | chat
    target_word: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None),
    ai_scoring: bool = Form(True),
    provider: str = Form("gemini"),
    scoring_provider: str = Form("gemini"),
    threshold: Optional[float] = Form(None),
    language: str = Form("português"),
    system: str = Form("Você é um assistente útil que responde de forma curta."),
):
    provider = (provider or "gemini").lower()
    scoring_provider = (scoring_provider or "gemini").lower()
    action = (action or "evaluate").lower()
    """
    🎯 Endpoint PRINCIPAL para avaliar a pronúncia com IA.
    
    **Fluxo:**
    1. Transcreve o áudio (Whisper local OU OpenAI/Gemini)
    2. Avalia com GPT/Gemini (feedback qualitativa detalhado)
    
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
    # Prepare audio file: accept multipart upload OR JSON with base64
    tmp_created = False
    audio_path = None
    # If multipart/form-data provided file
    if audio is not None:
        data = await audio.read()
        suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            audio_path = tmp.name
        tmp_created = True
    else:
        # try to parse JSON body for base64 audio
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                j = await request.json()
            except Exception:
                return JSONResponse({"error": "Corpo JSON inválido."}, status_code=400)

            # map fields if present in JSON
            user_id = user_id or j.get("user_id")
            action = j.get("action", action)
            target_word = target_word or j.get("target_word")
            ai_scoring = ai_scoring if ("ai_scoring" not in j) else (str(j.get("ai_scoring")).lower() in ["true", "1"])
            provider = j.get("provider", provider)
            scoring_provider = j.get("scoring_provider", scoring_provider)
            threshold = j.get("threshold", threshold)
            language = j.get("language", language)
            system = j.get("system", system)

            audio_b64 = None
            audio_name = "audio.wav"
            if isinstance(j.get("audio"), dict) and j["audio"].get("base64"):
                audio_b64 = j["audio"].get("base64")
                audio_name = j["audio"].get("name", audio_name)
            elif j.get("audio_base64"):
                audio_b64 = j.get("audio_base64")
                audio_name = j.get("audio_name", audio_name)

            if audio_b64:
                try:
                    decoded = base64.b64decode(audio_b64)
                except Exception:
                    return JSONResponse({"error": "Campo audio_base64 inválido (não é base64)."}, status_code=400)

                suffix = os.path.splitext(audio_name)[1] or ".wav"
                fd, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "wb") as f:
                    f.write(decoded)
                audio_path = path
                tmp_created = True

    if not audio_path:
        return JSONResponse({"detail": [{"type": "missing", "loc": ["body", "user_id"], "msg": "Field required", "input": None}, {"type": "missing", "loc": ["body", "audio"], "msg": "Field required", "input": None}]}, status_code=400)

    try:
        # Usa o arquivo salvo para transcrição (Gemini espera caminho de arquivo real)
        transcription = _transcrever_arquivo(audio_path, provider)
    except Exception as e:
        try:
            if tmp_created and audio_path and os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass
        return JSONResponse({"error": f"Falha na transcrição ({provider}): {e}"}, status_code=400)

    submission_id = "sub_" + uuid.uuid4().hex

    # ACTION: transcribe -> only transcription
    if action == "transcribe":
        try:
            return JSONResponse({
                "submission_id": submission_id,
                "transcription": transcription,
                "status": "done",
                "provider": provider,
            })
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass

    # ACTION: chat -> transcribe + chat reply
    if action == "chat":
        try:
            try:
                reply = _resposta_chat_texto(transcription, provider, system)
            except Exception as e:
                return JSONResponse({"error": f"Falha ao conversar com {provider}: {e}"}, status_code=400)

            return JSONResponse({
                "submission_id": submission_id,
                "transcription": transcription,
                "reply": reply,
                "provider": provider,
            })
        finally:
            try:
                os.remove(audio_path)
            except Exception:
                pass

    # ACTION: evaluate (default) -> transcribe + scoring
    try:
        if ai_scoring:
            score_result = pronunciation_score_with_ai(
                target_word,
                transcription,
                provider=scoring_provider,
                language=language,
            )
        else:
            score_result = pronunciation_score(target_word, transcription)
    finally:
        try:
            os.remove(audio_path)
        except Exception:
            pass

    # enrich result with common fields
    score_result["user_id"] = user_id
    score_result["transcription_provider"] = provider
    score_result["audio_name"] = audio.filename
    score_result["submission_id"] = submission_id
    score_result["transcription"] = transcription

    # compute pass if threshold provided and numeric score is present
    try:
        if threshold is not None and isinstance(score_result.get("score"), (int, float)):
            score_result["pass"] = float(score_result.get("score")) >= float(threshold)
    except Exception:
        pass

    # Garantir que o campo `match` exista (compatibilidade com método tradicional)
    if "match" not in score_result:
        score_result["match"] = bool(score_result.get("hit", False))

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

# -----------------------
# Catálogo de tarefas e gerador simples
# -----------------------
tasks_catalog = {
    "leitura_rapida": {
        "title": "Leitura Rápida / Fluência Verbal",
        "description": "Textos curtos (10–15 segundos) para avaliar velocidade, prosódia e clareza.",
        "expected_duration_s": 12,
        "instructions": "Leia o texto em voz alta de forma natural, sem pausas longas.",
        "samples": [
            "O rato roeu a roupa do rei de Roma.",
            "O sol nasceu e a cidade acordou.",
            "Hoje a escola terá aula de música e pintura."
        ],
        "suggested_threshold": 65
    },
    "trava_linguas": {
        "title": "Trava-línguas",
        "description": "Frases com alta complexidade articulatória; repetição rápida e precisa.",
        "expected_duration_s": 8,
        "instructions": "Repita o trava-línguas rapidamente mantendo clareza articulatória.",
        "samples": [
            "Três pratos de trigo para três tigres tristes",
            "O rato roeu a roupa do rei de Roma",
            "Pão com massa, passa a massa no pano"
        ],
        "suggested_threshold": 70
    },
    "frases_curtas": {
        "title": "Frases Curtas de Repetição / Leitura",
        "description": "Frases simples para avaliar memória auditiva e organização sintática.",
        "expected_duration_s": 5,
        "instructions": "Repita cada frase exatamente como ouvido ou leia em voz alta.",
        "samples": [
            "Ela abriu a janela.", "O menino comprou pão.", "Passa o sal, por favor."
        ],
        "suggested_threshold": 60
    },
    "repeticao_fonemas": {
        "title": "Repetição de Fonemas e Pares Mínimos",
        "description": "Contraste de fonemas e pares mínimos para discriminação e articulação.",
        "expected_duration_s": 6,
        "instructions": "Repita cada par claramente, com espaço entre as palavras.",
        "samples": [
            "papa / baba", "pato / bato", "sapo / xapo", "casa / caça"
        ],
        "suggested_threshold": 70
    },
    "leitura_palavras": {
        "title": "Leitura de Palavras e Pseudopalavras",
        "description": "Listas misturando palavras reais e pseudopalavras.",
        "expected_duration_s": 8,
        "instructions": "Leia a lista de palavras em voz alta, tentando manter ritmo constante.",
        "samples": [
            "gato, casa, pindó, maral, tromba", "festa, bico, lapor, suven"
        ],
        "suggested_threshold": 65
    },
    "repeticao_silabas": {
        "title": "Repetição de Sílabas",
        "description": "Sequências silábicas organizadas para controle articulatório e coordenação.",
        "expected_duration_s": 6,
        "instructions": "Repita a sequência rapidamente e de forma contínua.",
        "samples": [
            "pa pe pi po pu", "três tigres tristes", "pinga a pipoca na panela"
        ],
        "suggested_threshold": 60
    },
    "trava_linguas_progressiva": {
        "title": "Trava-línguas com Progressão Silábica",
        "description": "Combina sílabas repetitivas e frases difíceis com progressão de dificuldade.",
        "expected_duration_s": 10,
        "instructions": "Execute a progressão começando devagar e aumentando a velocidade mantendo clareza.",
        "samples": [
            "pa pe pi po pu - pa pe pi po pu - pa pe pi po pu",
            "três tigres tristes tricotando três tricôs"
        ],
        "suggested_threshold": 72
    }
}

def _extract_target_words(text: str, category: str):
	"""Heurística simples para extrair possíveis alvo(s) de cada item."""
	import re
	category = (category or "").lower()
	if category == "repeticao_fonemas":
		# pares separados por /
		if "/" in text:
			parts = [p.strip() for p in text.split("/")]
			return parts
		return [w.strip() for w in re.split(r"[,\s]+", text) if w.strip()]
	if category == "leitura_palavras":
		# palavras separadas por vírgula
		return [w.strip() for w in text.split(",") if w.strip()]
	if category == "repeticao_silabas":
		# retorna sílabas/words
		return [w.strip() for w in re.split(r"[,\s]+", text) if w.strip()]
	if category == "frases_curtas" or category == "leitura_rapida":
		# escolher palavras-chaves (substantivos/verbos) - heurística: words >3 chars
		words = [w.strip(".,") for w in text.split() if len(w.strip(".,") ) > 3]
		return words[:3] if words else [text]
	return [text]

def _generate_texts(category: str, count: int = 5, age_group: str = "adulto", difficulty: str = "medio", include_meta: bool = False):
    """
    Gerador simples sem IA para criar variações de itens por categoria.
    - agora suportando include_meta: quando True, retorna dicts com meta úteis.
    """
    import random
    if category not in tasks_catalog:
        raise ValueError("Categoria desconhecida")

    samples = tasks_catalog[category]["samples"]
    out = []

    # parâmetros simples para ajuste de comprimento e complexidade
    word_multiplier = 1
    if age_group == "infantil":
        word_multiplier = 1
    elif age_group == "juvenil":
        word_multiplier = 1.3
    else:
        word_multiplier = 1.6

    if difficulty == "facil":
        word_multiplier *= 0.9
    elif difficulty == "dificil":
        word_multiplier *= 1.2

    # Geração por categoria (regras simples)
    for i in range(count):
        item_text = ""
        if category == "leitura_rapida":
            parts = [random.choice(samples) for _ in range(max(1, int(word_multiplier)))]
            item_text = " ".join(parts)
        elif category == "repeticao_fonemas":
            p = random.choice(samples)
            if random.random() < 0.5:
                item_text = p
            else:
                a, b = p.split("/") if "/" in p else (p, p)
                item_text = f"{a.strip()} / {b.strip()}"
        elif category == "leitura_palavras":
            words = []
            for _ in range(max(4, int(4 * word_multiplier))):
                w = random.choice(random.choice(samples).split(","))
                words.append(w.strip())
            item_text = ", ".join(words)
        elif category == "frases_curtas":
            base = random.choice(samples)
            if random.random() < 0.5:
                item_text = base
            else:
                item_text = base + " " + random.choice(["Ela sorriu.", "Ele caminhou.", "O vento soprou."])
        elif category == "repeticao_silabas":
            if random.random() < 0.6:
                item_text = " ".join([random.choice(samples).split()[0] for _ in range(max(3, int(3 * word_multiplier)))])
            else:
                item_text = random.choice(samples)
        else:
            item_text = random.choice(samples)

        if include_meta:
            meta = {
                "text": item_text,
                "target_words": _extract_target_words(item_text, category),
                "instructions": tasks_catalog[category].get("instructions", ""),
                "estimated_duration_s": tasks_catalog[category].get("expected_duration_s", None)
            }
            out.append(meta)
        else:
            out.append(item_text)

    # garante unicidade simples
    seen = set()
    unique_out = []
    for t in out:
        # t pode ser dict ou str
        key = t["text"] if isinstance(t, dict) else t
        if key not in seen:
            unique_out.append(t)
            seen.add(key)
    return unique_out

# -----------------------
# Novos endpoints: listar e gerar tarefas
# -----------------------
@app.get("/tarefas")
async def listar_tarefas():
    """Retorna as categorias de tarefas e metadados (nome, descrição, número de exemplos)."""
    result = {
        k: {
            "title": v["title"],
            "description": v["description"],
            "sample_count": len(v.get("samples", []))
        }
        for k, v in tasks_catalog.items()
    }
    return JSONResponse(result)

@app.post("/tarefas/gerar")
async def gerar_tarefas(
    category: str = Form(...),                # chave da categoria (ex: leitura_rapida)
    count: int = Form(5),                     # quantos itens gerar
    age_group: str = Form("adulto"),          # infantil | juvenil | adulto
    difficulty: str = Form("medio"),          # facil | medio | dificil
    include_meta: bool = Form(False),         # se true, retorna objetos com meta (target_words, instructions...)
    use_ai: bool = Form(False),                # se true, pede ao Gemini/OpenAI para gerar os itens (JSON)
):
    """Gera N textos/itens para a categoria solicitada (pode usar IA se solicitado)."""
    category = (category or "").strip().lower()
    if category not in tasks_catalog:
        return JSONResponse({"error": "Categoria desconhecida", "available": list(tasks_catalog.keys())}, status_code=400)
    try:
        # If requested, try to delegate generation to the LLM (Gemini/OpenAI). We ask for strict JSON.
        if use_ai:
            system_prompt = (
                "Você é um gerador de itens pedagógicos. Receba uma categoria e gere exatamente {count} itens no formato JSON: "
                "[{\"text\": <texto>, \"instructions\": <instruções opcionais>}, ...]. Sempre retorne JSON válido sem explicações extras."
            )
            human = f"Categoria: {category}. Gere {count} itens curtos adequados para a categoria. Dê preferência a frases naturais em português."
            try:
                reply = _resposta_chat_texto(human, "gemini", system_prompt)
                import json
                try:
                    items = json.loads(reply)
                    # normalize: if items are strings convert to dicts
                    normalized = []
                    for it in items:
                        if isinstance(it, str):
                            normalized.append({"text": it})
                        elif isinstance(it, dict):
                            normalized.append(it)
                    return JSONResponse({
                        "category": category,
                        "title": tasks_catalog[category]["title"],
                        "age_group": age_group,
                        "difficulty": difficulty,
                        "items": normalized
                    })
                except Exception:
                    # LLM did not return proper JSON; fallback to builtin generator
                    pass
            except Exception:
                # if LLM call fails, continue to fallback
                pass

        texts = _generate_texts(category, count=count, age_group=age_group, difficulty=difficulty, include_meta=include_meta)
        return JSONResponse({
            "category": category,
            "title": tasks_catalog[category]["title"],
            "age_group": age_group,
            "difficulty": difficulty,
            "items": texts
        })
    except Exception as e:
        return JSONResponse({"error": f"Falha ao gerar tarefas: {e}"}, status_code=500)

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