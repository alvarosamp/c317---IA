"""
Script para iniciar o servidor FastAPI
Execute: python start_server.py
"""
import os
import sys

# Garantir que estamos no diretório correto
#os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configurar encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

# Adicionar o diretório atual ao PYTHONPATH
sys.path.insert(0, os.getcwd())

print("=" * 70)
print("🚀 INICIANDO SERVIDOR DE AVALIAÇÃO DE PRONÚNCIA")
print("=" * 70)
print(f"📁 Diretório: {os.getcwd()}")
print(f"🐍 Python: {sys.executable}")
print(f"🌐 URL: http://127.0.0.1:8000")
print(f"📚 Docs: http://127.0.0.1:8000/docs")
print("=" * 70)
print()

# Importar e rodar uvicorn
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
