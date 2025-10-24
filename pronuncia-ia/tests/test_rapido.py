"""
Teste Rápido - Sistema de Avaliação de Pronúncia com IA
Execute: python test_rapido.py
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Carregar .env
load_dotenv()

# Adicionar paths necessários
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "models"))

from app.core.scoring import pronunciation_score_with_ai

def test_pronunciation():
    print("=" * 70)
    print("🎤 TESTE DE AVALIAÇÃO DE PRONÚNCIA COM IA")
    print("=" * 70)
    
    # Verificar se a API key está configurada
    gemini_key = os.getenv("GEMINI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    print(f"DEBUG - GEMINI_API_KEY: {gemini_key[:20] if gemini_key else 'NÃO ENCONTRADA'}...")
    print(f"DEBUG - GOOGLE_API_KEY: {google_key[:20] if google_key else 'NÃO ENCONTRADA'}...")
    
    if not gemini_key and not google_key:
        print("❌ ERRO: Nenhuma chave API encontrada no arquivo .env")
        return
    
    print(f"✅ Chave da API Gemini encontrada!")
    print()
    
    # Teste 1: Pronúncia correta
    print("📝 Teste 1: Pronúncia CORRETA")
    print("-" * 70)
    texto_esperado = "Hello, how are you today?"
    texto_falado = "Hello, how are you today?"
    
    print(f"Texto esperado: {texto_esperado}")
    print(f"Texto falado:   {texto_falado}")
    print("\n⏳ Processando com Gemini AI...\n")
    
    try:
        resultado = pronunciation_score_with_ai(
            expected=texto_esperado,
            predicted=texto_falado,
            provider="gemini",
            language="en-US"
        )
        
        print("📊 RESULTADO:")
        print(f"  • Nota: {resultado['score']}/100")
        print(f"  • Match: {'✅ Correto' if resultado.get('match', False) else '❌ Incorreto'}")
        print(f"  • Método: {resultado.get('method', 'N/A')}")
        print(f"\n💬 Feedback:")
        print(f"  {resultado['feedback']}")
        
        if resultado.get('errors'):
            print(f"\n⚠️ Erros identificados:")
            for i, erro in enumerate(resultado['errors'], 1):
                print(f"  {i}. {erro}")
        
        if resultado.get('suggestions'):
            print(f"\n💡 Sugestões:")
            for i, sugestao in enumerate(resultado['suggestions'], 1):
                print(f"  {i}. {sugestao}")
        
    except Exception as e:
        print(f"❌ ERRO ao processar: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    
    # Teste 2: Pronúncia com erros
    print("\n📝 Teste 2: Pronúncia com ERROS")
    print("-" * 70)
    texto_esperado = "The weather is beautiful today"
    texto_falado = "The weater is butiful today"
    
    print(f"Texto esperado: {texto_esperado}")
    print(f"Texto falado:   {texto_falado}")
    print("\n⏳ Processando com Gemini AI...\n")
    
    try:
        resultado = pronunciation_score_with_ai(
            expected=texto_esperado,
            predicted=texto_falado,
            provider="gemini",
            language="en-US"
        )
        
        print("📊 RESULTADO:")
        print(f"  • Nota: {resultado['score']}/100")
        print(f"  • Match: {'✅ Correto' if resultado.get('match', False) else '❌ Incorreto'}")
        print(f"  • Método: {resultado.get('method', 'N/A')}")
        print(f"\n💬 Feedback:")
        print(f"  {resultado['feedback']}")
        
        if resultado.get('errors'):
            print(f"\n⚠️ Erros identificados:")
            for i, erro in enumerate(resultado['errors'], 1):
                print(f"  {i}. {erro}")
        
        if resultado.get('suggestions'):
            print(f"\n💡 Sugestões:")
            for i, sugestao in enumerate(resultado['suggestions'], 1):
                print(f"  {i}. {sugestao}")
        
        print("\n" + "=" * 70)
        print("✅ TESTES CONCLUÍDOS COM SUCESSO!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ ERRO ao processar: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pronunciation()
