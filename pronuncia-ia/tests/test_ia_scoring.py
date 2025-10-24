#!/usr/bin/env python3
"""
🧪 Script de teste para avaliação de pronúncia com IA
Execute: python test_ia_scoring.py
"""

import os
import sys
from pathlib import Path

# Adicionar paths necessários
sys.path.insert(0, str(Path(__file__).parent / "app" / "core"))
sys.path.insert(0, str(Path(__file__).parent / "models"))

from scoring import pronunciation_score, pronunciation_score_with_ai

def teste_basico():
    """Teste com método tradicional (Levenshtein)"""
    print("\n" + "="*60)
    print("🔧 TESTE 1: Método Tradicional (Levenshtein)")
    print("="*60)
    
    casos = [
        ("hello", "hello"),
        ("hello", "helo"),
        ("beautiful", "butiful"),
        ("pronunciation", "pronunsiation"),
    ]
    
    for esperado, falado in casos:
        resultado = pronunciation_score(esperado, falado)
        print(f"\n✏️  Esperado: '{esperado}'")
        print(f"🎤 Falado:   '{falado}'")
        print(f"📊 Score:    {resultado['score']}/100")
        print(f"✅ Match:    {resultado['hit']}")

def teste_com_ia():
    """Teste com GPT/Gemini"""
    print("\n" + "="*60)
    print("🤖 TESTE 2: Avaliação com IA (GPT/Gemini)")
    print("="*60)
    
    # Verificar se as APIs estão configuradas
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    gemini_ok = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    
    print(f"\n🔑 OpenAI configurada: {'✅ SIM' if openai_ok else '❌ NÃO'}")
    print(f"🔑 Gemini configurada: {'✅ SIM' if gemini_ok else '❌ NÃO'}")
    
    if not openai_ok and not gemini_ok:
        print("\n⚠️  AVISO: Nenhuma API configurada!")
        print("📝 Configure no arquivo .env (veja .env.example)")
        print("   GOOGLE_API_KEY=sua_chave  (Gemini - GRÁTIS)")
        print("   OPENAI_API_KEY=sua_chave  (GPT - Pago)")
        return
    
    # Escolher provider disponível
    provider = "gemini" if gemini_ok else "openai"
    
    print(f"\n🎯 Usando: {provider.upper()}")
    print("\n⏳ Aguarde, consultando IA...\n")
    
    casos = [
        ("hello", "hello", "inglês"),
        ("hello", "helo", "inglês"),
        ("beautiful", "butiful", "inglês"),
        ("olá mundo", "ola mundo", "português"),
    ]
    
    for esperado, falado, idioma in casos:
        print(f"\n{'─'*60}")
        print(f"✏️  Esperado: '{esperado}'")
        print(f"🎤 Falado:   '{falado}'")
        
        try:
            resultado = pronunciation_score_with_ai(
                esperado, 
                falado, 
                provider=provider,
                language=idioma
            )
            
            print(f"📊 Score:    {resultado['score']}/100")
            print(f"🎯 Método:   {resultado.get('method', 'N/A')}")
            print(f"\n💬 Feedback:")
            print(f"   {resultado.get('feedback', 'N/A')}")
            
            if resultado.get('errors'):
                print(f"\n❌ Erros detectados:")
                for erro in resultado['errors']:
                    print(f"   • {erro}")
            
            if resultado.get('suggestions'):
                print(f"\n💡 Sugestões:")
                for sugestao in resultado['suggestions']:
                    print(f"   • {sugestao}")
                    
        except Exception as e:
            print(f"❌ Erro: {e}")

def teste_comparacao():
    """Compara método tradicional vs IA"""
    print("\n" + "="*60)
    print("⚖️  TESTE 3: Comparação Tradicional vs IA")
    print("="*60)
    
    esperado = "beautiful"
    falado = "butiful"
    
    print(f"\n✏️  Esperado: '{esperado}'")
    print(f"🎤 Falado:   '{falado}'")
    
    # Método tradicional
    print("\n🔧 MÉTODO TRADICIONAL:")
    trad = pronunciation_score(esperado, falado)
    print(f"   Score: {trad['score']}/100")
    print(f"   Feedback: {trad.get('feedback', 'N/A')}")
    
    # Método com IA
    gemini_ok = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    
    if gemini_ok or openai_ok:
        provider = "gemini" if gemini_ok else "openai"
        print(f"\n🤖 MÉTODO COM IA ({provider.upper()}):")
        print("   ⏳ Aguarde...")
        
        try:
            ia = pronunciation_score_with_ai(esperado, falado, provider=provider)
            print(f"   Score: {ia['score']}/100")
            print(f"   Feedback: {ia.get('feedback', 'N/A')[:200]}...")
            
            print(f"\n📈 DIFERENÇA:")
            print(f"   Score IA vs Tradicional: {ia['score'] - trad['score']:+.1f} pontos")
            print(f"   Detalhamento: IA {'tem' if len(ia.get('feedback', '')) > 100 else 'não tem'} feedback rico")
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    else:
        print("\n⚠️  IA não disponível (configure .env)")

def main():
    print("\n" + "🎯"*30)
    print("   TESTE DE AVALIAÇÃO DE PRONÚNCIA COM IA")
    print("   Projeto C317 - Inteligência Artificial")
    print("🎯"*30)
    
    # Carregar .env se existir
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print("\n✅ Arquivo .env carregado!")
        else:
            print("\n⚠️  Arquivo .env não encontrado (usando variáveis do sistema)")
    except ImportError:
        print("\n💡 Dica: pip install python-dotenv (para carregar .env)")
    
    # Executar testes
    teste_basico()
    teste_com_ia()
    teste_comparacao()
    
    print("\n" + "="*60)
    print("✅ TESTES CONCLUÍDOS!")
    print("="*60)
    print("\n💡 PRÓXIMOS PASSOS:")
    print("   1. Configure suas chaves de API no .env")
    print("   2. Inicie a API: uvicorn app.api.main:app --reload")
    print("   3. Acesse: http://localhost:8000/docs")
    print("   4. Teste o endpoint /avaliar com áudio real!")
    print()

if __name__ == "__main__":
    main()
