"""
Teste da API REST - Sistema de Avaliação de Pronúncia
Execute: python test_api.py
"""
import requests
import json

print("=" * 70)
print("🧪 TESTANDO API REST - AVALIAÇÃO DE PRONÚNCIA")
print("=" * 70)
print()

base_url = "http://localhost:8000"

# ============================================================================
# TESTE 1: Health Check
# ============================================================================
print("1️⃣ TESTE: Health Check")
print("-" * 70)
try:
    response = requests.get(f"{base_url}/")
    print(f"✅ Status: {response.status_code}")
    print(f"📄 Resposta: {response.json()}")
except Exception as e:
    print(f"❌ Erro: {e}")
print()

# ============================================================================
# TESTE 2: Avaliação com IA - Pronúncia PERFEITA
# ============================================================================
print("2️⃣ TESTE: Avaliação com IA - Pronúncia PERFEITA")
print("-" * 70)
try:
    payload = {
        "expected": "Hello, how are you today?",
        "predicted": "Hello, how are you today?",
        "ai_scoring": True,
        "scoring_provider": "gemini",
        "language": "en-US"
    }
    
    response = requests.post(f"{base_url}/avaliar", json=payload)
    result = response.json()
    
    print(f"✅ Status: {response.status_code}")
    print(f"📊 Nota: {result.get('score')}/100")
    print(f"🎯 Match: {result.get('match')}")
    print(f"🔧 Método: {result.get('method')}")
    print(f"💬 Feedback: {result.get('feedback', '')[:200]}...")
    
    if result.get('suggestions'):
        print(f"💡 Sugestões: {len(result.get('suggestions'))} item(s)")
    
except Exception as e:
    print(f"❌ Erro: {e}")
print()

# ============================================================================
# TESTE 3: Avaliação com IA - Pronúncia COM ERROS
# ============================================================================
print("3️⃣ TESTE: Avaliação com IA - COM ERROS")
print("-" * 70)
try:
    payload = {
        "expected": "The weather is beautiful today",
        "predicted": "The weater is butiful today",
        "ai_scoring": True,
        "scoring_provider": "gemini",
        "language": "en-US"
    }
    
    response = requests.post(f"{base_url}/avaliar", json=payload)
    result = response.json()
    
    print(f"✅ Status: {response.status_code}")
    print(f"📊 Nota: {result.get('score')}/100")
    print(f"🎯 Match: {result.get('match')}")
    print(f"🔧 Método: {result.get('method')}")
    print(f"💬 Feedback: {result.get('feedback', '')[:200]}...")
    
    if result.get('errors'):
        print(f"⚠️ Erros identificados:")
        for erro in result.get('errors', []):
            print(f"   • {erro}")
    
    if result.get('suggestions'):
        print(f"💡 Sugestões ({len(result.get('suggestions'))} item(s)):")
        for i, sug in enumerate(result.get('suggestions', [])[:3], 1):
            print(f"   {i}. {sug}")
    
except Exception as e:
    print(f"❌ Erro: {e}")
print()

# ============================================================================
# TESTE 4: Método Tradicional (sem IA)
# ============================================================================
print("4️⃣ TESTE: Método Tradicional (Levenshtein)")
print("-" * 70)
try:
    payload = {
        "expected": "Hello world",
        "predicted": "Hello world",
        "ai_scoring": False
    }
    
    response = requests.post(f"{base_url}/avaliar", json=payload)
    result = response.json()
    
    print(f"✅ Status: {response.status_code}")
    print(f"📊 Nota: {result.get('score')}/100")
    print(f"🔧 Método: {result.get('method')}")
    print(f"💬 Feedback: {result.get('feedback')}")
    
except Exception as e:
    print(f"❌ Erro: {e}")
print()

# ============================================================================
# RESUMO
# ============================================================================
print("=" * 70)
print("✅ TESTES CONCLUÍDOS!")
print("=" * 70)
print()
print("🎯 O que você pode fazer agora:")
print("   1. Acessar documentação interativa: http://localhost:8000/docs")
print("   2. Ver documentação ReDoc: http://localhost:8000/redoc")
print("   3. Integrar com frontend/mobile")
print("   4. Adicionar transcrição de áudio real")
print()
print("📝 Lembre-se: Este sistema usa Gemini AI gratuitamente!")
print("   Limite: 60 requisições/minuto")
