# Dicionário de dados

Fonte: **Spotify Tracks Dataset**, de Maharshi Pandya
([Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset),
espelho no [Hugging Face](https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset)).
Atributos de áudio extraídos da API Web do Spotify; coleta de 2022.
Arquivo original: `dados/bruto/dataset.csv` (114.000 linhas, 21 colunas).
Arquivo limpo: `dados/processado/spotify_limpo.csv` (gerado por `preparar.py`).

**Uso no modelo:** A = alvo · E = entrada do modelo · G = agrupamento da divisão
treino/teste · X = removida da modelagem (identificador ou derivada).

## Colunas originais

| Coluna | Tipo | Descrição | Uso | Tratamento em `preparar.py` |
|---|---|---|:---:|---|
| `Unnamed: 0` | inteiro | índice da linha no arquivo original | X | removida |
| `track_id` | texto | identificador da faixa no Spotify | X | chave da consolidação: 1 linha por `track_id` |
| `artists` | texto | artista(s), separados por `;` | X | usado só para criar `musica_id` e `artista_principal` |
| `album_name` | texto | nome do álbum | X | — |
| `track_name` | texto | nome da faixa | X | usado só para criar `musica_id` |
| `popularity` | inteiro 0–100 | popularidade calculada pelo Spotify (reproduções recentes) | A | maior valor entre as linhas do mesmo `track_id`; `0` tratado como ausente (D3) |
| `duration_ms` | inteiro | duração em milissegundos | X | convertida em `duracao_min` |
| `explicit` | booleano | conteúdo explícito | E | convertido para 0/1 |
| `danceability` | real 0–1 | adequação para dança | E | — |
| `energy` | real 0–1 | energia/intensidade | E | — |
| `key` | inteiro 0–11 | tonalidade (0 Dó … 11 Si) | E | one-hot no pipeline |
| `loudness` | real (dB) | volume médio; mediana −7,2 dB | E | — |
| `mode` | 0/1 | modo (0 menor, 1 maior) | E | — |
| `speechiness` | real 0–1 | presença de fala (> 0,66 ≈ só fala) | E | — |
| `acousticness` | real 0–1 | probabilidade de ser acústica | E | — |
| `instrumentalness` | real 0–1 | probabilidade de não ter voz | E | — |
| `liveness` | real 0–1 | probabilidade de gravação ao vivo | E | — |
| `valence` | real 0–1 | positividade/alegria | E | — |
| `tempo` | real (BPM) | andamento estimado | E | `0` (157 faixas) → ausente, imputado pela mediana do treino |
| `time_signature` | inteiro | compasso estimado (3 a 7, em /4) | E | `0` e `1` (1.008 faixas) → ausente, imputado pelo mais frequente; one-hot |
| `track_genre` | texto | gênero (114 categorias) | E | faixas em vários gêneros: um gênero sorteado (semente 42); one-hot |

## Colunas criadas

| Coluna | Tipo | Descrição | Uso |
|---|---|---|:---:|
| `duracao_min` | real | `duration_ms / 60000` | E |
| `musica_id` | texto | artista + nome normalizados (minúsculas, espaços simples) | — |
| `grupo_id` | inteiro | une faixas com o mesmo `musica_id` **ou** com áudio idêntico (outro título: "feat.", "From …", tradução); a mesma música fica inteira no treino ou no teste | G |
| `artista_principal` | texto | primeiro artista de `artists`, normalizado | experimento de divisão por artista |
| `popularidade_zero` | 0/1 | 1 se `popularity == 0` | filtro (D3) |

## Números da limpeza

Ver `resultados/preparacao.json` e `docs/decisoes.md` (D1–D5).

| Etapa | Linhas |
|---|---:|
| CSV original | 114.000 |
| Sem ausentes em artista/álbum/nome | 113.999 |
| Sem linhas 100% duplicadas | 113.549 |
| Uma linha por `track_id` | 89.740 |
| Sem popularidade zero (modelo principal) | 80.393 |

## Características da amostra (limitações)

- **Amostragem por gênero, não aleatória:** o arquivo original tem exatamente
  1.000 faixas por gênero. A proporção de gêneros e a distribuição de popularidade **não
  representam o catálogo do Spotify**; o modelo descreve esta amostra.
- **Popularidade é um retrato de 2022** e muda com o tempo; não há data de
  lançamento no dataset.
- **Faixas que não são música** foram mantidas, porque pertencem a gêneros do
  próprio dataset: há 855 faixas de praticamente só fala (`speechiness` > 0,66)
  no conjunto do modelo, 764 delas do gênero `comedy`, além de sons
  ambientes no gênero `sleep`.
- **Mesma gravação, popularidades muito diferentes:** versões com áudio
  idêntico podem ter popularidade 75 e 1 (ex.: single × versão "feat."),
  o que é ruído irredutível no alvo.

## Licença

O cartão do dataset no Hugging Face indica licença **BSD**; um espelho do
Kaggle indica **CC0**. Ambas permitem redistribuição com crédito ao autor,
por isso o CSV original está incluído no repositório para facilitar a
replicação. Uso acadêmico, sem fins comerciais.
