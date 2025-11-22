Teste de integração rápida

Arquivo: test_evaluate_audio.py
Propósito: enviar o áudio `audioteste/audio.opus` para o endpoint `/avaliar` e mostrar a resposta.

Como usar:
1. Ative o venv do projeto:

   source /Users/alvarosamp/Documents/Projetos/p8/Top1/.venv/bin/activate

2. Instale requests se necessário:

   pip install requests

3. Inicie o servidor (em outra janela/terminal):

   python scripts/start_server.py

4. Rode o teste:

   python tests/test_evaluate_audio.py

Se o servidor estiver em outra URL, defina a variável de ambiente PRONUNCIACORE_SERVER antes de rodar.
