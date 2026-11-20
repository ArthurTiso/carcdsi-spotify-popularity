# Previsão de popularidade no Spotify com ML + LLM

**[Português](#português) · [English](#english)**

Projeto final da disciplina **CARCDSI** (IFSP). Sistema que estima a
popularidade (0–100) de uma música do Spotify a partir das características de
áudio e do gênero, explica cada previsão com **SHAP** e usa uma **LLM** (NVIDIA,
via SDK da OpenAI) para comunicar o resultado e para transformar uma descrição
em texto livre em uma previsão (*tool calling*).

| Aluno | Prontuário |
|---|---|
| Arthur de Morais Marques | CG3033856 |
| Vinícius Francisco Garcia Sobral | CG3034241 |

---

## Português

### Visão geral

```
dataset.csv ─► preparar.py ─► explorar.py ─► treinar.py ─► prever.py ─► processar.py / app.py
 (Kaggle)       limpeza        gráficos EDA   5 modelos +    ficha.json →   LLM explica e
                                              CV agrupada    previsão+SHAP  conversa
```

| Script | O que faz | Gera |
|---|---|---|
| `preparar.py` | limpeza: duplicatas, valores inválidos, agrupamento de versões da mesma música, marcação da popularidade zero | `dados/processado/spotify_limpo.csv`, `resultados/preparacao.json` |
| `explorar.py` | análise exploratória | `resultados/figuras/01..05_*.png`, `resultados/eda.json` |
| `treinar.py` | Dummy, média do gênero, Ridge, Random Forest e HistGradientBoosting; ajuste por validação cruzada agrupada (5 folds); métricas; importância por permutação; experimentos | `modelos/modelo.joblib`, `resultados/metricas.json`, `resultados/figuras/06..08_*.png` |
| `prever.py` | lê `ficha.json`, valida, prevê e explica com SHAP | `resultado.json` |
| `processar.py` | LLM: relato, chat e assistente com *tool calling* | — |
| `app.py` | interface Streamlit para demonstração | — |
| `config.py` | caminhos, colunas e semente compartilhados | — |

Decisões de metodologia (com justificativa e números) estão em
[`docs/decisoes.md`](docs/decisoes.md).

### Resultados

<!-- RESULTADOS_PT -->

### Como executar

Requisitos: **Python 3.14** e uma chave da NVIDIA (build.nvidia.com) para a parte
de LLM. O dataset original já está em `dados/bruto/dataset.csv`.

**Windows (PowerShell)** — execute uma linha por vez, na pasta do projeto:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe preparar.py
.\.venv\Scripts\python.exe explorar.py
.\.venv\Scripts\python.exe treinar.py
.\.venv\Scripts\python.exe prever.py
$env:NVIDIA_API_KEY = "nvapi-..."
.\.venv\Scripts\python.exe processar.py --modo relato
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Linux / macOS:**

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python preparar.py
.venv/bin/python explorar.py
.venv/bin/python treinar.py
.venv/bin/python prever.py
export NVIDIA_API_KEY="nvapi-..."
.venv/bin/python processar.py --modo relato
.venv/bin/python -m streamlit run app.py
```

Observações:

- `treinar.py` leva de 15 a 30 minutos (busca de hiperparâmetros). Para testar
  o código mais rápido: `treinar.py --rapido`. Para o experimento com a
  popularidade zero no modelo principal: `treinar.py --com-zeros`.
- `processar.py` tem três modos: `--modo relato` (explica `resultado.json`),
  `--modo chat` (relato + conversa) e `--modo assistente` (descreva a música
  em texto; a LLM chama o modelo de ML).
- Modelo de LLM padrão: `nvidia/nemotron-3.5-bolt-e4b`. Para trocar, defina
  `NVIDIA_MODEL` (ex.: `nvidia/nemotron-3.5-lightning-30b-a3b`). Para usar outro
  serviço compatível com a API da OpenAI, defina também `LLM_BASE_URL`.
- **Nunca** publique sua chave: ela fica só na variável de ambiente do terminal.

### Estrutura

```
├── config.py, preparar.py, explorar.py, treinar.py, prever.py, processar.py, app.py
├── ficha.json               # exemplo de entrada para prever.py
├── requirements.txt         # versões fixas
├── dados/
│   ├── bruto/dataset.csv    # original do Kaggle
│   ├── origem.json          # fonte e contagem de linhas
│   └── processado/          # gerado por preparar.py (não versionado)
├── modelos/                 # gerado por treinar.py (não versionado)
├── resultados/              # métricas (JSON) e figuras
└── docs/decisoes.md         # decisões de metodologia
```

### Dados

Spotify Tracks Dataset, de Maharshi Pandya, disponível no
[Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset):
114.000 linhas, 114 gêneros, atributos de áudio da API do Spotify.

---

## English

### Overview

Final project for the **CARCDSI** course (IFSP). The system estimates the
popularity (0–100) of a Spotify track from its audio features and genre,
explains every prediction with **SHAP**, and uses an **LLM** (NVIDIA, through
the OpenAI SDK) to communicate the result and to turn a free-text description
into a prediction (*tool calling*).

| Script | Purpose |
|---|---|
| `preparar.py` | data cleaning (duplicates, invalid values, grouping versions of the same song, flagging zero popularity) |
| `explorar.py` | exploratory data analysis (figures in `resultados/figuras/`) |
| `treinar.py` | Dummy, genre-mean, Ridge, Random Forest and HistGradientBoosting; grouped 5-fold cross-validation; test metrics; permutation importance; extra experiments |
| `prever.py` | reads `ficha.json`, validates it, predicts and explains with SHAP → `resultado.json` |
| `processar.py` | LLM: report, chat and tool-calling assistant |
| `app.py` | Streamlit demo interface |

Methodology decisions are documented (in Portuguese) in
[`docs/decisoes.md`](docs/decisoes.md).

### Results

<!-- RESULTADOS_EN -->

### How to run

Requirements: **Python 3.14** and an NVIDIA API key (build.nvidia.com) for the
LLM features. The raw dataset is already in `dados/bruto/dataset.csv`.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python preparar.py
.venv/bin/python explorar.py
.venv/bin/python treinar.py          # 15–30 min; use --rapido for a quick test
.venv/bin/python prever.py
export NVIDIA_API_KEY="nvapi-..."
.venv/bin/python processar.py --modo relato    # or: chat | assistente
.venv/bin/python -m streamlit run app.py
```

On Windows, use `.\.venv\Scripts\python.exe` instead of `.venv/bin/python` and
`$env:NVIDIA_API_KEY = "nvapi-..."` to set the key.

### Data

Spotify Tracks Dataset by Maharshi Pandya,
[Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset).
