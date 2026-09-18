# Guia de desenvolvimento — Músico Inteligente

## Objetivo

Aplicativo desktop local que escuta áudio, estima acordes/tom/andamento, acompanha
cifras e gera acompanhamento de baixo. Não usa serviços de IA remotos para analisar áudio.

## Execução

Instale Python e as dependências com `python -m pip install -r requirements.txt`.
Execute `python main.py` ou `iniciar.bat` no Windows. O ambiente validado localmente
usa Python 3.14.4. OCR requer Tesseract ou o mecanismo Windows disponível; importações
opcionais dependem dos pacotes descritos em `requirements.txt`.

## Mapa do código

- `app/audio`: decodificação, fontes de áudio e reprodução via sounddevice.
- `app/analysis`: pitch, chroma, acordes, tonalidade, BPM e análise estrutural.
- `app/music`: relógio, contexto musical, cifras, alinhamento, transposição e rastreamento.
- `app/instruments`: decisão, padrões, eventos e síntese do baixo virtual.
- `app/song`, `app/setlist`, `app/project`: repertório, sessões e persistência JSON.
- `app/input`: importação TXT/DOCX/PDF/imagem e parsing de cifras.
- `app/ui`: interface Tkinter, forma de onda e cromagrama.
- `tests`: testes unitários, de integração e regressão.

## Regras de integração

A reprodução usa blocos multicanais; a análise usa o sinal mono original, sem o
baixo sintetizado. O callback não deve ler novamente a fonte e avançar o cursor duas vezes.
A síntese estéreo deve ser convertida para mono quando a saída tem um único canal.

Atualizações de widgets devem ocorrer na thread Tkinter. A análise assíncrona de BPM
usa uma fila e uma geração de faixa para descartar resultados antigos.
O BPM manual (`bpm_override`) tem prioridade sobre o BPM detectado.
`SongSession.next_section` retorna um nome; `next_section_jump()` navega.

## Dados e salvamento

`virtual_band_project.json` contém o repertório local e fica fora do Git.
`MainWindow` recebe um caminho explícito e restaura o repertório salvo.
`ProjectManager()` sem caminho cria um projeto novo em memória; passe um caminho
para carregar um arquivo existente. Os testes devem sempre passar caminhos temporários
quando uma operação pode salvar dados.

O JSON é escrito em um temporário no mesmo diretório e substituído com `os.replace`.
Não apague o destino antes da substituição. Propague falhas de salvamento para a interface.
Uma alteração pode permanecer na memória após falha de gravação; permita tentar salvar novamente.

## Validação

Execute na raiz:

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
```

A versão inicial publicada passou em 193 testes, incluindo 15 regressões novas.
A suíte inclui um teste Tkinter e precisa de sessão gráfica; em Linux sem tela,
use um display virtual. A passagem dos testes não substitui audição e testes de hardware.
As amostras WAV incluídas são geradas pelo próprio projeto.

## Como propor mudanças

Descreva o problema e o comportamento esperado, indique os arquivos envolvidos e
faça mudanças pequenas. Para correções de integração, adicione uma regressão que
reproduza a falha, rode a suíte e descreva os resultados no commit ou pull request.
Não envie tokens, arquivos `.env`, repertórios pessoais ou cópias de segurança.
