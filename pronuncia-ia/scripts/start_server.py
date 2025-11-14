"""
Script para iniciar o servidor FastAPI
Execute: python start_server.py
"""
import os
import sys
import uvicorn
import pathlib

# Configurar encoding
os.environ["PYTHONIOENCODING"] = "utf-8"

# Garantir que o diretório raiz do projeto (pai da pasta `scripts`) esteja no PYTHONPATH
# Isso permite importar o pacote `app` independentemente de onde o script for executado.
project_root = pathlib.Path(__file__).resolve().parent.parent
try:
    # Mudar working directory para o root do projeto para comportamento previsível
    os.chdir(project_root)
except Exception:
    pass

sys.path.insert(0, str(project_root))

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


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
