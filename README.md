# Sistema de Avaliação de Pronúncia com IA

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Sistema inteligente para avaliação automática de pronúncia utilizando múltiplos modelos de Speech-to-Text (STT) e IA generativa. Esta versão dá prioridade à integração com o Gemini (Google) para transcrição e avaliação qualitativa.

## 📖 Sobre o Projeto

Este projeto implementa uma API REST que permite avaliar a qualidade da pronúncia de palavras através da comparação entre o texto esperado e o texto transcrito automaticamente do áudio fornecido pelo usuário.

### Características Principais

- **Múltiplos Modelos STT**: Suporte para Whisper, Wav2Vec2, DeepSpeech, Coqui STT e Faster Whisper
- **IA Generativa**: Integração com o modelo Gemini (Google) para análise, feedback e geração de relatórios personalizados
- **API REST**: Interface simples e eficiente com FastAPI
- **Algoritmo de Scoring**: Sistema de pontuação baseado na distância de Levenshtein
- **Testes Automatizados**: Cobertura completa de testes com pytest
- **Suporte CUDA**: Aceleração GPU para modelos compatíveis

## 🏗️ Arquitetura

```text
pronuncia-ia/
├── app/
│   ├── api/
│   │   └── main.py              # Endpoint principal da API
│   ├── core/
│   │   └── scoring.py           # Algoritmos de pontuação
│   ├── tests/
│   │   └── test_models.py       # Testes unitários
│   └── ui/                      # Interface do usuário (futuro)
├── models/
│   └── modelos.py               # Classes dos modelos STT
├── config/                      # Configurações
└── data/                        # Dados de treinamento/teste
```

## 🚀 Instalação

### Pré-requisitos

- Python 3.8+
- pip
- CUDA (opcional, para aceleração GPU)

### Passos de Instalação

1. **Clone o repositório**

   ```bash
   git clone https://github.com/alvarosamp/c317---IA.git
   cd c317---IA/IA
   ```

2. **Crie um ambiente virtual**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Instale as dependências**

   ```bash
   pip install -r requirements.txt
   ```

4. **Instale dependências específicas**

   ```bash
   pip install Levenshtein
   pip install fastapi uvicorn
   pip install transformers torch
   pip install librosa
   ```

## 💻 Uso

### Iniciando a API

```bash
cd pronuncia-ia/app/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

A API estará disponível em `http://localhost:8000`

### Endpoint Principal

**POST** `/avaliar`

Avalia a pronúncia de uma palavra fornecida em áudio.

**Parâmetros:**

- `user_id` (str): ID do usuário
- `target_word` (str): Palavra-alvo para avaliação
- `audio` (file): Arquivo de áudio (.wav, .mp3)

**Resposta:**

```json
{
    "score": 85.5,
    "similarity": 80.0,
    "hit": false,
    "predicted": "texto transcrito"
}
```

### Exemplo de Uso com curl

```bash
curl -X POST "http://localhost:8000/avaliar" \
  -F "user_id=user123" \
  -F "target_word=hello" \
  -F "audio=@audio_file.wav"
```

## 🧪 Testes

Execute os testes unitários:

```bash
cd pronuncia-ia/app/tests
pytest --maxfail=1 --disable-warnings -q
```

Para gerar relatório de cobertura:

```bash
pytest --cov=../core --cov=../../models --cov-report=html
```

## 🤖 Atualização e detalhes sobre IA (Gemini)

Nesta implementação o Gemini é o provedor padrão para duas etapas críticas:
1. Transcrição (quando configurado) — GeminiTranscriber
2. Avaliação qualitativa / feedback — GeminiChat via pronunciation_score_with_ai

Principais pontos:
- Gemini é usado por padrão para garantir consistência com a versão testada localmente.
- Ainda há suporte a OpenAI (whisper, chat) via classes OpenAITranscriber / OpenAIChat; escolha configurável por endpoint.
- Para ambientes com recursos limitados, há opção de transcrever localmente (Whisper) — porém no projeto padrão local o Whisper está desabilitado por RAM e Gemini é priorizado.

### Variáveis de ambiente e chaves
Configure suas chaves no ambiente antes de rodar:
- GEMINI_API_KEY — chave para acessar Gemini (se aplicável à integração)
- OPENAI_API_KEY — chave OpenAI (opcional, se usar OpenAI)
- OUTRAS — quaisquer variáveis exigidas por wrappers de modelo (ex.: PATH para modelos locais)

Exemplo (PowerShell):
```powershell
$env:GEMINI_API_KEY="sua_chave_gemini"
$env:OPENAI_API_KEY="sua_chave_openai"
```

### Fluxo de transcrição e avaliação
1. O endpoint recebe um upload via `multipart/form-data` com o campo `audio` (arquivo real).
2. O arquivo é salvo temporariamente no servidor (tempfile).
3. A função `_transcrever_arquivo(caminho_tmp, provedor)` chama o transcriber apropriado:
   - provedor == "gemini" → GeminiTranscriber.transcribe(caminho_tmp)
   - provedor == "openai" → OpenAITranscriber.transcribe(caminho_tmp)
   - Caso whisper fosse habilitado, poderia usar whisper_model.transcribe(...)
4. O texto transcrito é enviado para `pronunciation_score_with_ai(...)` que usa o chat LLM para gerar feedback detalhado, sugestões e score final.
5. Resposta JSON com score, feedback, highlights e metadados.

### Como forçar Gemini na API
- Padrão flexível (recomendado): parâmetros `provider` e `scoring_provider` com default `"gemini"`.
- Forçar no código (sempre usar Gemini): defina internamente
```python
provider = "gemini"
scoring_provider = "gemini"
```

### Exemplo de uso — Swagger (UI)
1. Acesse `http://127.0.0.1:8000/docs`
2. Trabalhe no endpoint `POST /avaliar`
3. Clique em "Try it out"
4. Preencha `user_id`, `target_word`, `ai_scoring` etc.
5. No campo `audio` clique em "Choose File" e selecione seu `.opus`/`.wav`
6. Execute (Execute) — atenção: o upload deve ser arquivo real (não JSON)

### Exemplo de uso — curl (multipart/form-data)
Enviar arquivo e usar Gemini:
```bash
curl -X POST "http://127.0.0.1:8000/avaliar" \
  -H "accept: application/json" \
  -F "user_id=user123" \
  -F "target_word=Testando" \
  -F "audio=@C:/caminho/para/teste.opus" \
  -F "ai_scoring=true" \
  -F "language=português"
```

Se você enviar JSON (Content-Type: application/json ou x-www-form-urlencoded) em vez de multipart/form-data, receberá 422 Unprocessable Entity — sempre use `-F` ou o upload do Swagger.

### Teste local (sem rodar a API)
Há um script `test_local.py` para validar GeminiTranscriber e a pipeline de scoring fora da API:
```bash
python test_local.py
```
Ajuste `audio_path` no topo do script para apontar ao seu arquivo local (`audioteste/teste.opus`) e defina `scoring_provider="gemini"` para replicar o comportamento da API.

### Logs e debug
- Rode o servidor com `--reload` para desenvolvimento:
  ```bash
  uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
  ```
- Verifique mensagens no terminal do uvicorn para erros de transcrição, chaves ausentes ou falhas de integração com Gemini/OpenAI.
- Ative prints ou logging no `modelos.py` e `scoring.py` para inspecionar payloads.

### Troubleshooting (erros comuns)
- 422 Unprocessable Entity: request não está em multipart/form-data com campo `audio` como arquivo.  
- "Field required" no Swagger: certifique-se de clicar em "Choose File" para `audio` e não colar JSON/objeto.  
- Arquivo vazio / transcrição vazia: confirme que `audio.read()` foi chamado apenas uma vez ou que salvou antes de passar para transcriber.  
- Erro de chave / 401: verifique variáveis de ambiente e permissões na conta do provedor.  
- GeminiTranscriber falhando localmente: teste com `test_local.py` e habilite logs no wrapper de modelo.

### Custos, limites e performance
- Gemini / OpenAI usage may incur costs. Teste com amostras curtas e monitore requests.
- Para produção, considere:
  - Limitar tamanho do upload
  - Queue/worker para processamento assíncrono
  - Cache de transcrições quando apropriado
  - Monitoramento e alertas para quotas

### Boas práticas de produção
- Não execute modelos pesados diretamente no servidor HTTP; use workers/process queue.
- Remova arquivos temporários imediatamente após uso (ex.: bloco finally com os.remove).
- Configure timeouts e retries para chamadas externas ao provedor.
- Habilite autenticação para os endpoints da API.

### Resposta típica do endpoint /avaliar
Exemplo de saída:
```json
{
  "score": 82.5,
  "similarity": 85.0,
  "feedback": "Boa entonação, ajuste no som /r/ final...",
  "suggestions": ["Pratique com minimal pairs: ...", "Use exercício X..."],
  "user_id": "user123",
  "transcription_provider": "gemini",
  "audio_name": "teste.opus"
}
```

## ⚙️ Execução rápida (recapitulando)
1. Ative venv:
   ```powershell
   .\.venv\Scripts\activate
   ```
2. Exporte chaves:
   ```powershell
   $env:GEMINI_API_KEY="sua_chave_gemini"
   ```
3. Rode:
   ```bash
   uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
   ```
4. Teste no Swagger `/docs` (upload do arquivo como arquivo).

## 📝 Contato e suporte
- Abra uma issue no repositório ou envie e-mail para suporte@exemplo.com

## � Como Citar

Se este projeto foi útil para sua pesquisa ou trabalho, cite da seguinte forma:

```
@misc{pronuncia-ia,
   author = {Álvaro Sampaio and Diego Rodrigues and Pedro Bressan},
   title = {Sistema de Avaliação de Pronúncia com IA},
   year = {2025},
   howpublished = {\url{https://github.com/alvarosamp/c317---IA}}
}
```

## �🔗 Links Úteis

- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [Whisper OpenAI](https://openai.com/research/whisper)
- [Transformers Hugging Face](https://huggingface.co/transformers/)
- [Gemini (Google AI)](https://deepmind.google/technologies/gemini/)

## 📞 Suporte

Para dúvidas e suporte, abra uma issue no repositório ou envie um e-mail para:

suporte@exemplo.com

Ou utilize o e-mail institucional dos desenvolvedores.