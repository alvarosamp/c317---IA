# 🎙️ Pronuncia-IA - API de Avaliação de Pronúncia

API FastAPI para avaliação de pronúncia usando modelos de IA (Gemini, GPT, Whisper).

## 🚀 Início Rápido

```powershell
# 1. Ativar ambiente virtual
cd c:\Users\vish8\OneDrive\Desktop\p8\C317\c317---IA
.\.venv\Scripts\Activate.ps1

# 2. Instalar dependências
cd pronuncia-ia
pip install -r requirements.txt

# 3. Configurar .env
cp .env.example .env
# Editar .env e adicionar sua GEMINI_API_KEY

# 4. Iniciar servidor
python scripts/start_server.py
```

## 📂 Estrutura do Projeto

```
pronuncia-ia/
├── app/                    # Código principal da API
│   ├── api/                # Endpoints FastAPI
│   │   └── main.py         # 5 endpoints (/avaliar, /transcrever, etc)
│   ├── core/               # Lógica de negócio
│   │   ├── scoring.py      # Algoritmos de avaliação
│   │   └── storage.py      # Gerenciamento de arquivos
│   └── tests/              # Testes unitários
├── models/                 # Modelos de IA
│   ├── modelos.py          # Classes STT e LLM
│   └── cuda.py             # Suporte GPU
├── docs/                   # 📚 Documentação completa
├── tests/                  # 🧪 Scripts de teste
├── scripts/                # ⚙️ Scripts utilitários
├── config/                 # Configurações
├── data/                   # Dados de teste
├── .env                    # Variáveis de ambiente
└── requirements.txt        # Dependências Python
```

## 🔗 Integração com Backend NestJS

**Veja documentação completa:** [../c317-backend/INTEGRACAO_IA.md](../../c317-backend/INTEGRACAO_IA.md)

### Como funciona:
```
Frontend → NestJS → process_audio.py → FastAPI (porta 8000) → Resposta
```

## 📚 Documentação

| Arquivo | Descrição |
|---------|-----------|
| [00_LEIA_PRIMEIRO.md](docs/00_LEIA_PRIMEIRO.md) | Introdução ao projeto |
| [COMO_TESTAR.md](docs/COMO_TESTAR.md) | Guia completo de testes |
| [STATUS_ATUAL.md](docs/STATUS_ATUAL.md) | Status e configuração |
| [PARA_O_PROFESSOR.md](docs/PARA_O_PROFESSOR.md) | Documentação acadêmica |
| [DECISOES_TECNICAS.md](docs/DECISOES_TECNICAS.md) | Justificativas técnicas |
| [EXPERIMENTOS_REALIZADOS.md](docs/EXPERIMENTOS_REALIZADOS.md) | Testes realizados |

## 🧪 Como Testar

### Teste 1: Direto (sem servidor)
```powershell
cd tests
python test_rapido.py
```

### Teste 2: API completa
```powershell
# Terminal 1: Iniciar servidor
python scripts/start_server.py

# Terminal 2: Testar endpoints
cd tests
python teste_api_simples.py
```

### Teste 3: Integração com Backend
```powershell
# Ver: c317-backend/INTEGRACAO_IA.md
```

## 🎯 Endpoints Disponíveis

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/avaliar` | POST | Avaliação completa (STT + IA scoring) |
| `/transcrever` | POST | Apenas transcrição de áudio |
| `/falar` | POST | Áudio → conversa com IA |
| `/chat_texto` | POST | Chat de texto com IA |
| `/tutor_pronuncia` | POST | Tutor interativo |

## ⚙️ Configuração (.env)

```env
# API Keys
GEMINI_API_KEY=sua_chave_aqui
OPENAI_API_KEY=sua_chave_aqui  # Opcional

# Modelos
GEMINI_MODEL=gemini-2.5-flash
DEFAULT_PROVIDER=gemini        # whisper | openai | gemini
DEFAULT_LANGUAGE=pt-BR

# Servidor
PORT=8000
HOST=0.0.0.0
```

## 🔧 Tecnologias

- **FastAPI** - Framework web
- **Google Gemini 2.5-flash** - LLM para avaliação (GRATUITO, 60 req/min)
- **Whisper** - STT local (modelo base ~100MB)
- **OpenAI Whisper API** - STT cloud (opcional)
- **Levenshtein** - Algoritmo de distância (fallback)

## 📊 Modelos Testados

O arquivo `scoring.py` contém comentários extensos mostrando os modelos testados:
- ✅ **Gemini 2.5-flash** (escolhido - gratuito, eficaz)
- ⚠️ Wav2Vec2 (muito lento, alta memória)
- ⚠️ Faster Whisper (complexidade desnecessária)
- ⚠️ Ensemble (overhead de manter múltiplos modelos)
- ✅ Levenshtein (simples, rápido, fallback confiável)

## 🎓 Projeto Acadêmico

**Disciplina:** C317 - Inteligência Artificial  
**Objetivo:** Sistema de avaliação de pronúncia com IA  
**Abordagem:** Comparação entre métodos tradicionais e LLMs

### Para o Professor
Ver [docs/PARA_O_PROFESSOR.md](docs/PARA_O_PROFESSOR.md) para:
- Justificativas técnicas das escolhas
- Experimentos realizados
- Comparação de abordagens
- Resultados obtidos

## 🚀 Deploy

```powershell
# 1. Clonar e configurar
git clone <repo>
cd pronuncia-ia
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configurar .env
cp .env.example .env
# Adicionar GEMINI_API_KEY

# 3. Iniciar
python scripts/start_server.py
```

## 📝 Licença

Projeto acadêmico - C317 2025
