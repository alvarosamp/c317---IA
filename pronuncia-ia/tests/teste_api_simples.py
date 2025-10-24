"""
Script Simples de Teste da API - Avaliação de Pronúncia
Execute depois que o servidor estiver rodando
"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("🧪 TESTANDO API DE AVALIAÇÃO DE PRONÚNCIA")
print("=" * 70)
print()

# Teste 1: Health Check
print("1️⃣ Testando conexão com servidor...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"   ✅ Servidor OK! Status: {response.status_code}")
    print(f"   Resposta: {response.json()}")
except Exception as e:
    print(f"   ❌ Erro: {e}")
    print("   💡 Certifique-se que o servidor está rodando!")
    exit(1)

print()

# Teste 2: Avaliação com IA - Pronúncia Perfeita
print("2️⃣ Testando avaliação com IA (Pronúncia PERFEITA)...")
try:
    payload = {
        "expected": "Hello, how are you today?",
        "predicted": "Hello, how are you today?",
        "ai_scoring": True,
        "scoring_provider": "gemini",
        "language": "en-US"
    }
    
    response = requests.post(f"{BASE_URL}/avaliar", json=payload)
    result = response.json()
    
    print(f"   ✅ Resposta recebida!")
    print(f"   📊 Nota: {result.get('score')}/100")
    print(f"   🎯 Match: {'✅ Correto' if result.get('match') else '❌ Incorreto'}")
    print(f"   🔧 Método: {result.get('method')}")
    print(f"   💬 Feedback: {result.get('feedback', 'N/A')[:100]}...")
    
except Exception as e:
    print(f"   ❌ Erro: {e}")

print()

# Teste 3: Avaliação com IA - Pronúncia com Erros
print("3️⃣ Testando avaliação com IA (Pronúncia COM ERROS)...")
try:
    payload = {
        "expected": "The weather is beautiful today",
        "predicted": "The weater is butiful today",
        "ai_scoring": True,
        "scoring_provider": "gemini",
        "language": "en-US"
    }
    
    response = requests.post(f"{BASE_URL}/avaliar", json=payload)
    result = response.json()
    
    print(f"   ✅ Resposta recebida!")
    print(f"   📊 Nota: {result.get('score')}/100")
    print(f"   🎯 Match: {'✅ Correto' if result.get('match') else '❌ Incorreto'}")
    print(f"   🔧 Método: {result.get('method')}")
    print(f"   💬 Feedback: {result.get('feedback', 'N/A')[:150]}...")
    
    if result.get('errors'):
        print(f"   ⚠️ Erros encontrados:")
        for erro in result['errors']:
            print(f"      • {erro}")
    
    if result.get('suggestions'):
        print(f"   💡 Sugestões:")
        for sug in result['suggestions'][:3]:
            print(f"      • {sug}")
    
except Exception as e:
    print(f"   ❌ Erro: {e}")

print()

# Teste 4: Método Tradicional (sem IA)
print("4️⃣ Testando método tradicional (Levenshtein)...")
try:
    payload = {
        "expected": "Hello world",
        "predicted": "Hello world",
        "ai_scoring": False
    }
    
    response = requests.post(f"{BASE_URL}/avaliar", json=payload)
    result = response.json()
    
    print(f"   ✅ Resposta recebida!")
    print(f"   📊 Nota: {result.get('score')}/100")
    print(f"   🔧 Método: {result.get('method')}")
    print(f"   💬 Feedback: {result.get('feedback')}")
    
except Exception as e:
    print(f"   ❌ Erro: {e}")

print()
print("=" * 70)
print("✅ TESTES CONCLUÍDOS!")
print("=" * 70)
print()
print("🌐 Para testar interativamente, acesse:")
print("   📚 Documentação: http://localhost:8000/docs")
print("   📖 ReDoc: http://localhost:8000/redoc")
