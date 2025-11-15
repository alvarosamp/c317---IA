import json
_lev_source = None
try:
    # Prefer python-Levenshtein (fast C implementation)
    from Levenshtein import distance as lev
    _lev_source = "python-Levenshtein"
except Exception:
    try:
        # Fallback to rapidfuzz if available
        from rapidfuzz.distance import Levenshtein as _rlev
        def lev(a, b):
            return _rlev.distance(a, b)
        _lev_source = "rapidfuzz"
    except Exception:
        # Final fallback: use difflib (pure-Python, slower and returns ratio -> convert to distance)
        import difflib
        def lev(a, b):
            if not a and not b:
                return 0
            ratio = difflib.SequenceMatcher(None, a, b).ratio()
            # Convert similarity ratio to an integer distance approximating Levenshtein
            return int(round((1.0 - ratio) * max(len(a), len(b))))
        _lev_source = "difflib"

print(f"[DEBUG] Using Levenshtein implementation: {_lev_source}")

import os
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env.
# Tentamos múltiplos locais para facilitar diferentes formas de executar o servidor
# 1) ./pronuncia-ia/.env (diretório do serviço)
# 2) ./c317---IA/.env (repo root - um nível acima)
# 3) cwd/.env (current working dir)
possible_envs = [
    Path(__file__).parent.parent.parent / ".env",  # pronuncia-ia/.env
    Path(__file__).parent.parent.parent.parent / ".env",  # repo root .env
    Path.cwd() / ".env",
]

env_path = None
for p in possible_envs:
    try:
        if p.exists():
            env_path = p
            break
    except Exception:
        continue

if env_path:
    load_dotenv(dotenv_path=env_path)
    print(f"[DEBUG] Carregado .env a partir de: {env_path}")
else:
    print(f"[DEBUG] Nenhum .env encontrado em: {[str(p) for p in possible_envs]}")

# Import dos modelos de chat
models_path = Path(__file__).parent.parent.parent / "models"
sys.path.insert(0, str(models_path))

print(f"[DEBUG] 📁 Path para models: {models_path}")
print(f"[DEBUG] 📁 Arquivo modelos.py existe: {(models_path / 'modelos.py').exists()}")

try:
    from modelos import OpenAIChat, GeminiChat
    print(f"[DEBUG] ✅ Import dos modelos bem sucedido!")
    print(f"[DEBUG] OpenAIChat: {OpenAIChat}")
    print(f"[DEBUG] GeminiChat: {GeminiChat}")
except ImportError as e:
    print(f"[DEBUG] ❌ Erro ao importar modelos: {e}")
    import traceback
    traceback.print_exc()
    OpenAIChat = None
    GeminiChat = None

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip() if ch.isalnum() or ch == " ")

def string_similarity(expected: str, predicted: str) -> float:
    """Cálculo de similaridade usando Levenshtein (método tradicional)"""
    a, b = _norm(expected), _norm(predicted)
    if not a and not b:
        return 1.0
    d = lev(a, b)  # Distância de Levenshtein
    return 1.0 - d / max(len(a), len(b))

# ============================================================================
# MÉTODOS ALTERNATIVOS TESTADOS (Modelos Pré-treinados Especializados)
# ============================================================================
# Durante o desenvolvimento, testamos diferentes abordagens para avaliação:
#
# 1. Análise Acústica com Wav2Vec2:
#    - Modelo especializado em reconhecimento de padrões de áudio
#    - Vantagem: Melhor para detectar nuances de pronúncia em português
#    - Desvantagem: Requer muito processamento e análise de features acústicas
#
# def pronunciation_score_wav2vec2(expected: str, predicted: str, audio_features) -> dict:
#     """
#     Avaliação usando features acústicas do Wav2Vec2.
#     Analisa diretamente as características do áudio além da transcrição.
#     """
#     from models.modelos import Wav2Vec2
#     model = Wav2Vec2()
#     # Análise de features acústicas (MFCCs, pitch, energia)
#     # acoustic_score = analyze_acoustic_features(audio_features)
#     # phonetic_accuracy = compare_phonemes(expected, predicted)
#     # return combined_score
#     pass
#
# 2. Análise com Faster Whisper (otimizado):
#    - Versão otimizada do Whisper com melhor performance
#    - Vantagem: Mais rápido que Whisper original
#    - Desvantagem: Ainda focado em transcrição, não em avaliação pedagógica
#
# def pronunciation_score_faster_whisper(expected: str, audio_path: str) -> dict:
#     """
#     Usa Faster Whisper para transcrição e análise de confiança.
#     """
#     from models.modelos import FasterWhisper
#     model = FasterWhisper()
#     # segments com scores de confiança por palavra
#     # confidence_scores = get_word_confidence(segments)
#     # return detailed_analysis
#     pass
#
# 3. Ensemble de Modelos:
#    - Combinar múltiplos modelos STT para consenso
#    - Vantagem: Mais robusto, reduz erros individuais
#    - Desvantagem: 3-5x mais lento, complexidade aumentada
#
# def pronunciation_score_ensemble(expected: str, audio_path: str) -> dict:
#     """
#     Combina resultados de múltiplos modelos para consenso.
#     """
#     # whisper_result = Whisper().transcribe(audio_path)
#     # wav2vec_result = Wav2Vec2().transcribe(audio_path)
#     # faster_whisper_result = FasterWhisper().transcribe(audio_path)
#     # consensus = voting_mechanism([whisper, wav2vec, faster])
#     # return aggregate_score(consensus)
#     pass
#
# CONCLUSÃO DA ANÁLISE:
# Optamos por usar LLMs (GPT/Gemini) pois oferecem:
# ✅ Feedback qualitativo rico e pedagógico (não apenas numérico)
# ✅ Identificação contextual de erros (entende o "porquê")
# ✅ Sugestões personalizadas de melhoria
# ✅ Análise linguística além da similaridade textual
# ✅ Menor complexidade de implementação
# ============================================================================

def pronunciation_score(expected: str, predicted: str) -> dict:
    """
    Calcula a pontuação de pronúncia baseado na similaridade entre a palavra-alvo e o texto reconhecido.
    MÉTODO TRADICIONAL (Levenshtein) - usado como fallback.
    """
    sim = string_similarity(expected, predicted)  # 0..1
    hit = 1.0 if _norm(expected) == _norm(predicted) else 0.0  # Verifica se é uma correspondência exata
    score = 0.8 * sim + 0.2 * hit  # A pontuação final, ponderando a similaridade e o hit
    return {
        "score": round(100 * score, 1),
        "similarity": round(100 * sim, 1),
        "hit": bool(hit),
        "predicted": predicted,
        "feedback": "Avaliação automática baseada em similaridade textual.",
        "method": "levenshtein"
    }

def pronunciation_score_with_ai(expected: str, predicted: str, provider: str = "openai", language: str = "português") -> dict:
    """
    Avalia pronúncia usando GPT/Gemini para análise qualitativa detalhada.
    
    Args:
        expected: Palavra/frase que deveria ser falada
        predicted: O que foi realmente transcrito
        provider: "openai" ou "gemini"
        language: Idioma para contextualizar a avaliação
    
    Returns:
        dict com score, feedback detalhado, sugestões, etc.
    """
    
    print(f"\n[DEBUG] 🚀 Iniciando avaliação com IA")
    print(f"[DEBUG] Provider: {provider}")
    print(f"[DEBUG] Language: {language}")
    print(f"[DEBUG] Expected: '{expected}'")
    print(f"[DEBUG] Predicted: '{predicted}'")
    
    # Validar se as classes estão disponíveis
    print(f"[DEBUG] OpenAIChat disponível: {OpenAIChat is not None}")
    print(f"[DEBUG] GeminiChat disponível: {GeminiChat is not None}")
    
    if provider.lower() == "openai" and OpenAIChat is None:
        print("[DEBUG] ⚠️ OpenAI não disponível, usando método tradicional")
        return pronunciation_score(expected, predicted)  # Fallback para método tradicional
    if provider.lower() == "gemini" and GeminiChat is None:
        print("[DEBUG] ⚠️ Gemini não disponível, usando método tradicional")
        return pronunciation_score(expected, predicted)  # Fallback para método tradicional
    
    # Prompt otimizado para avaliação de pronúncia
    prompt = f"""Você é um professor de {language} especializado em avaliação de pronúncia.

**TAREFA:** Avaliar a pronúncia do aluno comparando o que ele deveria falar com o que realmente foi transcrito.

**Palavra/Frase esperada:** "{expected}"
**O que foi transcrito:** "{predicted}"

**INSTRUÇÕES:**
1. Dê uma nota de 0 a 100 considerando:
   - Precisão das palavras (70%)
   - Possíveis erros de pronúncia detectados na transcrição (20%)
   - Clareza e fluência (10%)

2. Se a transcrição for EXATAMENTE igual ao esperado, dê nota 100.

3. Forneça feedback construtivo e específico:
   - O que o aluno acertou
   - Quais erros foram cometidos
   - Dicas práticas para melhorar

4. Se houver erros, identifique quais sons/palavras foram problemáticos.

**IMPORTANTE:** Retorne APENAS um JSON válido neste formato exato:
{{
    "score": <número de 0 a 100>,
    "match": <true se transcrição == esperado, false caso contrário>,
    "feedback": "<feedback detalhado em {language}>",
    "errors": ["<lista de erros específicos>"],
    "suggestions": ["<dicas práticas para melhorar>"],
    "highlights": {{
        "correct": ["<palavras/sons que acertou>"],
        "incorrect": ["<palavras/sons que errou>"]
    }}
}}

NÃO adicione texto antes ou depois do JSON. Retorne apenas o objeto JSON."""

    try:
        print(f"[DEBUG] 📝 Criando instância do chat {provider}...")
        
        # Chamar o modelo apropriado
        if provider.lower() == "gemini":
            chat = GeminiChat()
            print(f"[DEBUG] ✅ GeminiChat instanciado com sucesso")
        else:  # openai é o padrão
            chat = OpenAIChat()
            print(f"[DEBUG] ✅ OpenAIChat instanciado com sucesso")
        
        print(f"[DEBUG] 🤖 Enviando prompt para IA...")
        response_text = chat.reply_from_text(prompt, system="Você é um avaliador de pronúncia preciso. Sempre retorne JSON válido.")
        print(f"[DEBUG] 📨 Resposta recebida (primeiros 200 chars): {response_text[:200]}...")
        
        # Tentar extrair JSON da resposta (alguns modelos podem adicionar markdown)
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            print(f"[DEBUG] 🔧 Removendo marcador ```json")
            response_text = response_text[7:]
        if response_text.startswith("```"):
            print(f"[DEBUG] 🔧 Removendo marcador ```")
            response_text = response_text[3:]
        if response_text.endswith("```"):
            print(f"[DEBUG] 🔧 Removendo marcador ``` do final")
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        print(f"[DEBUG] 🔍 Tentando parsear JSON...")
        # Parse do JSON
        result = json.loads(response_text)
        print(f"[DEBUG] ✅ JSON parseado com sucesso!")
        print(f"[DEBUG] Score retornado: {result.get('score')}")
        
        # Garantir que tem todos os campos necessários
        return {
            "score": result.get("score", 0),
            "match": result.get("match", False),
            "predicted": predicted,
            "expected": expected,
            "feedback": result.get("feedback", "Sem feedback disponível."),
            "errors": result.get("errors", []),
            "suggestions": result.get("suggestions", []),
            "highlights": result.get("highlights", {"correct": [], "incorrect": []}),
            "method": f"ai-{provider}",
            "language": language
        }
        
    except json.JSONDecodeError as e:
        # Se falhar no parse JSON, retornar método tradicional
        print(f"[DEBUG] ❌ Erro ao parsear JSON da IA: {e}")
        print(f"[DEBUG] Resposta completa: {response_text}")
        fallback = pronunciation_score(expected, predicted)
        fallback["ai_response"] = response_text  # Para debug
        return fallback
        
    except Exception as e:
        # Qualquer outro erro, retornar método tradicional
        print(f"[DEBUG] ❌ Erro ao avaliar com IA: {type(e).__name__}: {e}")
        import traceback
        print(f"[DEBUG] Traceback completo:")
        traceback.print_exc()
        return pronunciation_score(expected, predicted)
