# Sistema de Avaliação de Pronúncia com IA

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

Sistema inteligente para avaliação automática de pronúncia utilizando múltiplos modelos de Speech-to-Text (STT) e algoritmos de similaridade textual.

## 📖 Sobre o Projeto

Este projeto implementa uma API REST que permite avaliar a qualidade da pronúncia de palavras através da comparação entre o texto esperado e o texto transcrito automaticamente do áudio fornecido pelo usuário.

### Características Principais

- **Múltiplos Modelos STT**: Suporte para Whisper, Wav2Vec2, DeepSpeech, Coqui STT e Faster Whisper
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

## 🤖 Modelos Suportados

### 1. **Whisper** (Padrão)

- Modelo: `openai/whisper-small`
- Características: Alta precisão, multilíngue
- Uso: Ideal para uso geral

### 2. **Wav2Vec2**

- Modelo: `jonatasgrosman/wav2vec2-large-xlsr-53-portuguese`
- Características: Otimizado para português
- Uso: Melhor para áudio em português

### 3. **DeepSpeech**

- Características: Leve, rápido
- Uso: Cenários com recursos limitados

### 4. **Coqui STT**

- Características: Open source, personalizável
- Uso: Implementações customizadas

### 5. **Faster Whisper**

- Características: Versão otimizada do Whisper
- Uso: Melhor performance em produção

## 📊 Sistema de Pontuação

O sistema utiliza uma combinação de métricas:

- **Similaridade**: Baseada na distância de Levenshtein (0-100%)
- **Match Exato**: Bonificação para correspondência perfeita
- **Score Final**: `0.8 × similaridade + 0.2 × match_exato`

## 🛠️ Desenvolvimento

### Estrutura de Desenvolvimento

1. **Adicionando novos modelos**: Implemente uma nova classe em `models/modelos.py`
2. **Novos algoritmos de scoring**: Adicione em `app/core/scoring.py`
3. **Testes**: Crie testes correspondentes em `app/tests/`

### Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 👥 Equipe

- **Desenvolvedoes**: Álvaro Sampaio, Diego Rodrigues, Pedro Bressan
- **Curso**: C317 - Inteligência Artificial

## 🔗 Links Úteis

- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [Whisper OpenAI](https://openai.com/research/whisper)
- [Transformers Hugging Face](https://huggingface.co/transformers/)

## 📞 Suporte

Para dúvidas e suporte, abra uma issue no repositório ou entre em contato através do email institucional.