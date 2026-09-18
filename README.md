# 🎸 Virtual Band AI — Laboratório de Audição & Contexto Musical

> **Objetivo do Projeto**: Construir uma banda virtual autônoma que escuta um instrumento real (violão, guitarra ou teclado), compreende a harmonia e o ritmo em tempo real e toca junto, iniciando por um **Baixista Virtual Autônomo** (v0.2+).

---

## 🎯 Princípio Fundamental

O sistema opera sob a premissa de **Zero Entrada Manual**:
- ❌ Não requer que o usuário informe previamente o tom.
- ❌ Não requer que o usuário informe o BPM.
- ❌ Não requer que o usuário cadastre cifras ou acordes.
- ❌ Não requer que o usuário informe o instrumento utilizado.

**Tudo é processado 100% localmente no computador via DSP (Processamento Digital de Sinais) e MIR (Music Information Retrieval)**, sem uso de APIs externas, serviços de nuvem ou modelos generativos pesados.

---

## 🏗️ Arquitetura do Sistema

```text
Entrada de Áudio (WAV / MP3 / Futuro Microfone ou Linha)
       ↓
Audio Analyzer
       ├→ Pitch Detection (Autocorrelação f0, Nota dominante e Cents)
       ├→ Chroma Extractor (Cromagrama de 12 classes C..B)
       ├→ Harmonic / Chord Detection (Matching de tríades e inversões no baixo)
       ├→ Key Detection (Krumhansl-Schmuckler adaptativo com leaky integrator)
       └→ Tempo / Beat Detection (Envelope de onsets e grade de batidas)
       ↓
Musical Context Manager (Memória Temporal e Estabilização)
       ├→ ChordStabilizer & ChordHistory (Debounce / Histerese e eventos discretos)
       ├→ KeyStabilizer & KeyHistory (Rastreamento persistente de tonalidade)
       └→ MusicalClock (Sincronizado com o áudio: BPM, Compasso, Beat, Posição)
       ↓
MusicalContext (Barramento de Dados Compartilhado)
       ↓
Futuros Músicos Virtuais (BassPlayer, Drummer, Keys, Guitar)
```

---

## 📦 Componentes Implementados

### 1. `MusicalContext` (`app/music/musical_context.py`)
Estrutura central que representa o estado musical instantâneo e temporal:
- **Nota Dominante**: `note`, `frequency`, `note_confidence`, `cents_deviation`.
- **Harmonia & Acorde Atual**: `chord`, `previous_chord`, `chord_start_time`, `chord_duration`, `chord_confidence`, `inversion`, `bass_note`, `detected_notes`.
- **Tonalidade (Key)**: `key`, `previous_key`, `key_start_time`, `key_duration`, `key_confidence`.
- **Relógio Musical (MusicalClock)**: `bpm`, `meter`, `bar` (compasso), `beat` (tempo 1 a 4), `beat_position` (fase [0..1)), `is_beat` (pulso).
- **Latência Quadripartida**: `processing_latency`, `analysis_window`, `stabilization_delay`, `estimated_musical_latency`.

### 2. `ChordHistory` e Estabilizador Temporal (`app/analysis/chord_history.py`)
- **Filtro de Ruído & Histerese**: Evita oscilações transitórias (ex: se um acorde oscilar `C → G → C`, o `G` de 1 frame é rejeitado como ruído).
- **Eventos Discretos (`ChordEvent`)**: Registra apenas quando uma transição harmônica real é confirmada (ex: `C` de 0.0s a 2.0s = 1 único evento com duração de 2.0s).
- **Tabela / Timeline**: Exibição visual de `TIME`, `CHORD`, `DURATION`, `CONFIDENCE`.

### 3. `KeyHistory` e Estabilizador de Tom (`app/analysis/key_history.py`)
- **Memória Deslizante (*Leaky Integrator*)**: Integra o cromagrama acumulado ao longo do tempo.
- **Rejeição de Falsas Modulações**: Requer persistência temporal de pelo menos 2.0s para confirmar uma mudança de tonalidade real, rejeitando acordes de passagem ou empréstimos modais.

### 4. `MusicalClock` (`app/music/musical_clock.py`)
- **Sincronismo Estrito com o Áudio**: O relógio calcula o compasso e tempo diretamente a partir do timestamp do áudio, respondendo instantaneamente a Seek, Pause, Play e Stop.
- **Métricas Flexíveis**: Suporta `4/4`, `3/4`, `2/4`, `6/8` etc.

### 5. Configurações Centralizadas (`app/music/constants.py`)
- `MIN_CHORD_CONFIDENCE = 0.40`
- `MIN_CHORD_STABILITY_TIME = 0.15` (150 ms)
- `CHORD_CHANGE_CONFIRMATION_TIME = 0.20` (200 ms)
- `MIN_KEY_CONFIDENCE = 0.60`
- `KEY_CHANGE_CONFIRMATION_TIME = 2.00` (2.0 s)
- `DEFAULT_METER = "4/4"`
- `DEFAULT_CHUNK_SIZE = 4096`

---

## ⏱️ Métricas de Latência Quadripartida

1. **`processing_latency`**: Tempo real gasto pela CPU no cálculo dos algoritmos DSP (medido em nanossegundos via `perf_counter` ~ 1.0 a 1.5 ms).
2. **`analysis_window`**: Tamanho do bloco temporal de áudio analisado (4096 amostras a 44100 Hz = 92.8 ms).
3. **`stabilization_delay`**: Atraso temporal configurado para confirmar a persistência do novo acorde e rejeitar ruídos (~ 200 ms).
4. **`estimated_musical_latency`**: Latência musical estimada total de ponta a ponta:
   $$\text{Latência Estimada} = \text{processing\_latency} + \frac{\text{analysis\_window}}{2} + \text{stabilization\_delay} \approx 245\text{ ms}$$

---

## 🚀 Como Iniciar no Windows

### 1. Instalar Dependências:
```powershell
pip install -r requirements.txt
```
O núcleo (áudio/DSP/MIR) exige apenas `numpy`, `scipy`, `soundfile`, `sounddevice` e `librosa`.
Os demais pacotes são opcionais e só são necessários para importar cifras em `.docx`, `.pdf`
ou imagem (OCR). Sem eles, o app roda normalmente e apenas o formato correspondente fica
indisponível — com uma mensagem clara indicando qual pacote instalar.

### 2. Modo Mais Rápido:
Dê um duplo clique no arquivo **`iniciar.bat`** na pasta do projeto.

### Ou via Linha de Comando:
```powershell
python main.py
```

---

## 🎸 Como Testar com Áudio de Violão Acústico

1. Abra a aplicação (via `iniciar.bat` ou `python main.py`).
2. Clique no botão marrom **`🎸 Violão Acústico (C-G-Am-F 120 BPM)`**:
   - Uma faixa modelada com violão acústico (timbre rico, 4 batidas por compasso, rasqueado de cordas e harmônicos reais) será carregada.
3. Clique em **`▶ PLAY`** (ou pressione a barra de espaço):
   - **Nota Atual**: acompanha a frequência predominante e desvio em cents.
   - **Acorde Atual**: exibe a cifra (`C`, `G`, `Am`, `F`), tempo no acorde (ex.: `1.85 s`), acorde anterior e inversões.
   - **Tonalidade**: estabiliza em `C Major`.
   - **Relógio Musical**: exibe `120 BPM`, `Compasso: X`, `Tempo: Y / 4` e pulso sincronizado nos LEDs `[● ○ ○ ○]`.
   - **Tabela Chord History**: preenche automaticamente a timeline com cada acorde finalizado, sua duração e confiança.

Você também pode abrir qualquer arquivo seu de violão ou teclado em formato **WAV ou MP3** pelo botão **`📂 ABRIR ÁUDIO`**.

---

## 🧪 Suíte de Testes Automatizados (210 Testes)

Execute todos os testes unitários e de integração com o comando:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

### Cobertura dos Testes:
- `test_audio.py`: E/S de WAV/MP3, envelopes de waveform, seek, tolerância a silêncio.
- `test_pitch.py`: Detectores de autocorrelação e HPS em múltiplas frequências.
- `test_chords.py`: Tríades maiores, menores, inversões e histórico.
- `test_tempo.py`: Extração de BPM, batidas e tonalidade Krumhansl-Schmuckler.
- `test_pipeline_simulation.py` / `test_pipeline_run.py`: Pipeline completo ponta a ponta.
- `test_musical_context.py`: Requisitos formais de contexto, debounce, 3/4, 100 BPM e relógio.
- `test_guitar_audio.py`: Validação sobre áudio acústico de violão.
- `test_key_stability.py`: Estabilidade de tonalidade e rejeição de falsas modulações.
- `test_long_audio_history.py`: Histórico em faixas longas (> 48 s).
- `test_chart_import.py` / `test_chart_editing.py`: Importação e edição de cifras (TXT, DOCX, PDF, imagem).
- `test_music_structure.py` / `test_position_tracking.py`: Estrutura musical e rastreamento de posição.
- `test_bass_player.py` / `test_musician_system.py`: Baixista virtual e sistema de músicos.

> As dependências opcionais de importação de cifra (`python-docx`, `pypdf`, `reportlab`)
> não são obrigatórias: os testes que dependem delas são automaticamente ignorados (`skip`)
> quando o pacote não está instalado.

---

## 🧠 Inteligência de Audição & Cifra (v0.5)

Melhorias recentes para ler cifras e casar o áudio com mais robustez:

1. **Classificação contextual de linhas de cifra**: uma linha só de acordes preserva TODOS
   os acordes, mesmo os que coincidem com palavras comuns (os acordes `A`, `E`, `B`).
   A desambiguação por stopwords só age em linhas ambíguas (mistura de acorde e letra).
2. **Vocabulário estendido do "ouvido"** (`ChordDetector(detect_extensions=True)`): além de
   tríades, reconhece **dominantes (b7)** e **suspensos (sus2/sus4)** de forma confiável.
3. **Detecção automática de transposição / capotraste** (`TranspositionTracker`): se a
   execução está deslocada por um número constante de semitons em relação à cifra, o sistema
   reconhece isso (em vez de reportar uma cascata de "erros") e expõe o deslocamento.
4. **Reancoramento por progressão**: quando um acorde isolado diverge mas a **sequência**
   recente ainda casa com um trecho da cifra, a posição é reancorada — mais robusto que o
   casamento de acorde único.
5. **Cifra como *prior* da audição** (`AudioAnalyzer.set_harmonic_expectation`): o ouvido
   deixa de adivinhar no escuro — recebe o acorde esperado da cifra e o tom como viés,
   desempatando casos ambíguos (ex.: `Cmaj7` vs `Em`). O viés é pequeno: só decide empates,
   **nunca sobrepõe** uma evidência de áudio forte, e **não infla** a confiança reportada
   (que continua baseada na similaridade bruta). No modo livre (sem cifra), a audição é cega
   como antes. Já ligado ao play-along na interface, realimentado a cada frame.
6. **Cromagrama por soma harmônica + Alta Resolução** (`AudioAnalyzer.set_harmonic_resolution`):
   um cromagrama que atribui cada parcial ao seu fundamental, **suprimindo o vazamento de
   overtones** (ex.: o Si-fantasma do 3º harmônico do Mi num acorde de Dó maior cai de ~0.4
   para ~0.14). Isso **destrava o reconhecimento de tétrades** (`maj7`, `m7`, `7`, `sus`),
   antes impossível. A detecção de TOM continua no cromagrama FFT clássico (calibração
   validada); apenas a de ACORDE usa o harmônico. Ligado automaticamente no play-along.
   O CQT (`librosa.cqt`) foi avaliado e **descartado**: em blocos de 4096 amostras (tempo
   real) degrada os graves e custa ~10× mais — a soma harmônica é superior neste regime.

## 🎼 Padrão CifraClub, Transposição e Seções (v0.6)

Baseado no padrão de cifras do **CifraClub** (a referência do app):

- **Cifragem CifraClub**: reconhece `C4`=sus4, `C2`=sus2, `Cm5-`/`C5-`=diminuto, `C7M`=7ª maior,
  `C5+`=aumentado, baixo invertido `G/B`, além de `7, 9, 11, 13` e alterações `C7(9)`.
- **Cabeçalhos reais**: `[Primeira/Segunda/Terceira Parte]`→Verso, `[Pré-Refrão]`, `[Refrão]`,
  `[Pós-Refrão]`/`[Refrão Final]`→Refrão, `[Solo]`, `[Ponte]`, `[Tab - …]`, `[Final]`→Outro.
- **Capotraste** (`Capotraste na 2ª casa`): o app entende que os acordes escritos são *shapes*
  e o áudio soa `N` semitons acima; usa isso para **localizar o músico pelo som real**
  (`PositionEstimator.set_capo`).
- **Tom** e **BPM/Andamento** extraídos automaticamente do cabeçalho.
- **Tablatura ignorada**: linhas de tab (`E|--3-5h7--|`, slides `/\\`, hammer `h`, pull `p`,
  bend `b`, `~`, `(5)`…) são reconhecidas e **excluídas da harmonia**, sem gerar acordes falsos
  e **sem fundir blocos** (um `[Solo]` só de tab não engole o `[Verso]` seguinte).
- **Cabeçalho inline**: `[Intro]  G#  C  Fm  C#9` (acordes na mesma linha do rótulo) é
  separado corretamente em seção + linha de acordes.
- **Sustenidos preservados**: acordes maiores como `G#`, `C#`, `F#` não são mais truncados
  (`G#`→`G`) — bug de fronteira de regex corrigido.
- **Heurística de espaçamento**: uma linha cujos acordes estão isolados ou separados por
  2+ espaços (alinhados às sílabas) é reconhecida como linha de acordes; letras com palavras
  minúsculas permanecem letra, mesmo espaçadas.
- Validado contra uma **amostra diversa de cifras reais** (gospel, sertanejo, rock, MPB,
  forró, pop, internacional) com testes de regressão de corpus.

**Transposição de tom** (`ChordChart.transpose_to_key`, `ProjectManager.transpose_song`):
escolha um novo tom e todos os acordes se adequam, preservando qualidade/extensão/inversão e
re-grafando em sustenidos ou bemóis conforme o tom-alvo. Atualiza acordes, letra-âncora,
texto exibido e a linha "Tom:".

**Auto-detecção de seções + cores** (`sectionizer.auto_sectionize_text`): cifras sem `[ ]` são
segmentadas e rotuladas (Intro/Verso/Refrão/Ponte…) por repetição de letra e progressão. Um mapa
`SECTION_COLORS` dá uma cor por tipo para a interface diferenciar os blocos visualmente.

**Operações de bloco** (`ChordChart.duplicate_section / move_section / remove_section`): copiar um
bloco inteiro já separado, reordenar e remover — base para o editor e para salvar no setlist/projeto.

## 🎯 Localização por áudio (seguir o músico, não a ordem da cifra)

Antes, a posição avançava só pelo **relógio** (percorria a cifra em ordem). Agora o **áudio
decide onde na cifra o músico está**: o `PositionEstimator` mantém um deslocamento
relógio→cifra e, ao reconhecer a **progressão recente de acordes**, procura no documento
inteiro onde ela encaixa e **reancora** a posição ali (`localize()` / `_global_localize()`).

- O relógio dá o avanço suave; o áudio corrige o *lugar*.
- **Casa pela TÔNICA (raiz)**: o detector às vezes erra a qualidade (maior/menor/7ª), mas
  acerta o baixo/tônica. Casamento exato vale 1.0, só pela tônica vale 0.8 — assim o sistema
  **localiza e reacha a música** mesmo com qualidades imperfeitas, e **confirma pela tônica**
  em vez de reportar falsa divergência.
- Um salto real exige **3+ acordes** casando; evidência curta/ambígua não salta
  (a cifra continua soberana no acorde isolado).
- Em seções repetidas (ex.: refrão que volta), prefere o encaixe **mais próximo** da posição
  atual, evitando pular para um refrão idêntico distante.
- Respeita o **capotraste** (compara o som real com os shapes escritos).
- Pode ser desligado (`_follow_audio = False`) para voltar ao avanço puramente temporal.

## 🖥️ Interface: cifra colorida por blocos, transposição e edição de blocos

No visualizador de cifra (aba **Músico Play-Along & Cifra**):
- **Blocos coloridos por tipo**: cada cabeçalho de seção vira um "chip" colorido —
  Intro (cinza), Verso (azul), Pré-Refrão (ciano), Refrão (vermelho), Ponte (roxo),
  Solo (laranja), Instrumental (verde-azulado), Outro (cinza-escuro) — para o músico
  diferenciar as partes num relance.
- **🎚 Tom**: um seletor transpõe a cifra inteira ao vivo (todos os acordes + a linha "Tom:").
- **⧉ Copiar bloco / ▲ / ▼**: duplica o bloco sob o cursor e reordena os blocos, preservando
  o alinhamento acorde-sobre-letra; a mudança é salva na música (setlist/projeto).

## ⚠️ Limitações Atuais Conhecidas (Transparência Técnica)

1. **Ataques Fortes de Palheta em Violão Aço**:
   Batidas percussivas de palheta produzem ruído branco transitório de alta amplitude nos primeiros 10-20 ms do ataque. O `ChordStabilizer` foi calibrado para tolerar esse transitório sem oscilar, mas em afinações muito abertas com cordas soltas ressoando por muito tempo, a nota do baixo pode capturar a 5ª ou 3ª da corda solta como inversão temporária.
2. **Ambiguidade de Oitava Rítmica em Levadas Esparsas**:
   Se um violonista toca apenas 2 batidas por compasso (no tempo 1 e tempo 3) em uma música a 120 BPM, o detector de transientes pode estimar 60 BPM (meio-tempo). O `MusicalClock` permite ajustar a fórmula de compasso ou andamento dinamicamente.
3. **Microfone / Entrada ao Vivo**:
   O `LiveAudioSource` está arquitetado como contrato, mas a captura em tempo real por microfone será refinada nas fases subsequentes (atualmente validado em arquivos WAV/MP3 e streaming local).
4. **Ambiguidade de raiz em acordes suspensos**:
   Um conjunto como `{C, F, G}` é simultaneamente `Csus4` e `Fsus2` (mesmas classes de nota). Sem a nota do baixo clara ou o prior da cifra, a escolha da tônica é ambígua. O prior harmônico (cifra) resolve a maioria dos casos no play-along.
5. **7ª maior / 7ª da menor sem cifra**:
   No modo livre (sem cifra), a detecção de `maj7`/`m7` depende do modo de alta resolução harmônica estar ativo. Com cifra ativa (play-along), o modo é ligado automaticamente e essas tétrades são reconhecidas de forma confiável.


## Desenvolvimento e colaboração

Leia [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) para entender a arquitetura,
os cuidados com dados locais e como validar mudanças. A suíte atual contém 210 testes.
O repertório `virtual_band_project.json` é local e não é enviado ao GitHub.
A interface restaura esse arquivo ao iniciar; sem ele, cria músicas de demonstração.

### Correções de integração e persistência

- Salvamento usa substituição atômica; falhas preservam o JSON anterior.
- Erros ao salvar são informados e o editor mantém a indicação de alterações pendentes.
- Baixo virtual é misturado tanto em áudio mono quanto estéreo.
- Trocar para música sem áudio libera a fonte anterior.
- BPM detectado atualiza a sessão, respeitando o ajuste manual.
- Nome da próxima seção e navegação têm APIs distintas: `next_section` e `next_section_jump()`.
- Análise de BPM ocorre em segundo plano; resultados de faixas anteriores são descartados.
- Testes que salvam repertórios usam diretórios temporários.
