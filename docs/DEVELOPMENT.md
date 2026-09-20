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

A suíte atual passou em 249 testes, incluindo 17 regressões da posição musical, 6 do seguimento por notas, 14 do tempo adaptativo, 12 do baixo preditivo e 7 da máquina de performance.
A suíte inclui um teste Tkinter e precisa de sessão gráfica; em Linux sem tela,
use um display virtual. A passagem dos testes não substitui audição e testes de hardware.
As amostras WAV incluídas são geradas pelo próprio projeto.

## Como propor mudanças

Descreva o problema e o comportamento esperado, indique os arquivos envolvidos e
faça mudanças pequenas. Para correções de integração, adicione uma regressão que
reproduza a falha, rode a suíte e descreva os resultados no commit ou pull request.
Não envie tokens, arquivos `.env`, repertórios pessoais ou cópias de segurança.

## Posição musical consolidada

O relógio representa tempo bruto. A localização oficial é produzida por
`PositionEstimator` e `ChartAlignment`, consolidada em `ChartPosition` e publicada
por `SongSession`. Consulte `current_bar`/`current_beat` para a música e
`clock_bar`/`clock_beat` para diagnóstico temporal.

A convenção existente é `musical_bar = clock_bar - bar_offset`: relógio 20 e
offset -16 representam compasso musical 36; relógio 21 representa 37. Perda
de detecção preserva o offset. Recuperações locais e globais persistem o ajuste.
Troca de música e reset eliminam o offset e o histórico da sessão anterior.

`AudioAnalyzer` entrega o contexto de áudio à sessão antes de despachar os
instrumentos. O `MusicalContext` publicado contém compasso, tempo, fase, seção,
linha, confiança e tracking do snapshot oficial; mantém relógio e offset em campos
de diagnóstico. Estrutura, memória e predição recebem essas coordenadas.
O registry reutiliza o baixo conectado ao mixer, evitando um segundo baixo.
A UI consome o snapshot para HUD, Follow Mode e scroll.

`ChartAlignment.align_from_clock` e o modo independente de `ContextManager`
continuam auxiliares sem cifra/reancoragem. Não use esses caminhos para localizar
uma sessão com cifra. A estimativa descritiva de estrutura não calcula offsets.

As regressões em `tests/test_position_source_of_truth.py` cobrem ausência de
offset, reancoragens para frente/para trás, continuidade sem áudio, recuperação
local e por progressão, troca, reset, transposição, edição, navegação, baixo,
memória, predição, diagnóstico e Follow Mode, incluindo o despacho integrado.

## Seguimento por notas do instrumento de referência

O `AudioAnalyzer` já extrai nota dominante e confiança. A `SongSession` passa essa
evidência ao mesmo `PositionEstimator` responsável pelo alinhamento dos acordes.
Notas são comparadas por classe de altura com as notas dos acordes soantes da
cifra (inclusive capotraste). Tônica e demais notas do acorde têm pesos
diferentes; uma nota estranha pode ser nota de passagem, sem forçar erro.

O estimador reúne eventos de notas distintas e confiáveis. Após pelo menos
três eventos, compara caminhos que permanecem no acorde ou avançam ao próximo
compasso. Só altera `_bar_offset` quando um trecho distante vence a posição
atual e as alternativas por margem suficiente. Passagens repetidas ou ambíguas
não causam salto. Eventos muito antigos, troca de música, reset e edição da
cifra limpam a sequência. O relógio permanece a base do avanço entre evidências.

Uma cifra descreve acordes, não a melodia exata. Por isso o alinhamento por
notas é probabilístico e pode se abster; ensaios reais com áudio de referência
são necessários para calibrar pesos e limiares por instrumento.

## Andamento adaptativo e agendamento do baixo

O `OnsetTempoDetector` oferece timestamps exatos da grade extraída da faixa
quando ela está disponível. Sem grade, observa ataques energéticos do sinal
corrente; um `is_beat` calculado da grade sintética nunca é usado como pulso
observado. O `MusicalClock` existente contém o seguidor de tempo/fase: o primeiro
ataque apenas arma uma hipótese, intervalos plausíveis ajustam `target_bpm`
por média exponencial ponderada pela confiança, e `current_bpm` converge a ela
com constante de tempo de 1,25 s. Um PLL simples corrige parcialmente o erro
de fase a cada pulso. Eventos isolados discrepantes não alteram o BPM; após
perda da detecção, o relógio avança em `HOLDOVER` e recupera gradualmente.

A `SongSession` publica o relógio no `MusicalContext` sem alterar a
responsabilidade de localização do `PositionEstimator`. No modo de sessão,
`BassPlayer` calcula a próxima batida e agenda o evento com tempo absoluto
no `BassSynthesizer`; o callback de áudio insere o sinal no sample correto
e descarta eventos vencidos. No limite do compasso, usa o próximo acorde
previsto na cifra. A antiga API imediata permanece para contextos avulsos.
O status de debug mostra BPM inicial/atual/alvo, confiança, fase, erro de
fase e estado.

A observação simples de energia em blocos não substitui um detector de
onsets especializado para entrada ao vivo polifônica, e o caminho de análise
da UI ainda depende da cadência de atualização Tkinter. Testes com interface
de áudio e instrumento real são necessários para medir latência de ponta a
ponta, falsos ataques e calibrar os limiares.

## Agendamento preditivo do baixo pela cifra

A `ChartPosition` já informa o próximo acorde e quantos compassos faltam para
a mudança. A `SongSession` publica no `MusicalContext` o compasso/tempo dessa
mudança, seção seguinte, confiança de tracking e uma geração da posição. O
`BassPlayer` consulta esse look-ahead e prepara a decisão da próxima batida;
o `BassSynthesizer` mantém uma fila pequena com nota, horário, origem,
confiança e geração. Somente o callback de áudio executa a nota no horário
agendado. Um acorde sustentado não muda um compasso cedo.

Quando o `PositionEstimator` reancora, a cifra é alterada ou o transporte busca
outro timestamp, a geração avança
e os eventos futuros da geração anterior são descartados sem cortar a nota
que já está soando. Uma mudança de BPM pode corrigir o horário do mesmo
evento ainda pendente. A fusão continua dando prioridade à cifra: um acorde
detectado isolado não altera a previsão. Uma divergência sustentada, confirmada
pela `ChartAudioFusion` com confiança de áudio suficiente, pode orientar os
próximos tempos; na virada do compasso a cifra retoma a prioridade, exceto
quando a localização está incerta e sua confiança caiu.

O agendamento conhece mudanças em fronteiras de compassos porque o modelo
atual de `ChartAlignment` ainda representa a duração de cada acorde em
compassos inteiros. Mudanças de acorde dentro do compasso e subdivisões
rítmicas exigem ampliar a linha temporal da cifra.

## Máquina de performance

`PerformanceState` pertence à `SongSession` e decide se a banda deve tocar;
ele não substitui `TrackingState` do `PositionEstimator` nem `HOLDOVER` do
relógio. A sequência é `WAITING → RECOVERING → PLAYING`; silêncio observado
por 2,5 segundos passa para `HOLDING`, e quatro segundos de pausa passam para
`WAITING`. Nessas fases a sessão preserva cifra, seção e última posição
confiável, mas o baixo cancela somente ataques futuros.

Após atividade retornar, o estimador volta a receber evidências e a banda só
prepara uma entrada no próximo downbeat. Fim natural exige silêncio prolongado
perto do final da cifra; silêncio no meio nunca vira `ENDED`. Ao terminar,
eventos futuros são cancelados e a sessão segue disponível para `start()` e
replay. A atividade vem do RMS do áudio, nota/acorde confiáveis ou pulso
observado; chamadas sem sinal de áudio não são tratadas como silêncio.
