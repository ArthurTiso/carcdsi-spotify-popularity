# Registro de decisões do projeto

Decisões técnicas e a justificativa de cada uma. Este arquivo serve de base
para a seção de Metodologia e de Discussão do artigo.

## D1 · Duplicatas por `track_id` (preparar.py)

- **Fato:** o CSV tem 114.000 linhas, mas só 89.741 `track_id` únicos. A mesma
  faixa aparece em vários gêneros (16.299 faixas em mais de um gênero). Entre
  as linhas do mesmo `track_id`, só `track_genre` e `popularity` mudam; todos os
  atributos de áudio são idênticos.
- **Decisão:** manter **uma linha por `track_id`**. O gênero é sorteado entre os
  rótulos da faixa (semente 42) e a popularidade é a **maior** registrada (as
  diferenças vêm do momento da coleta; 720 faixas têm valores diferentes).
- **Por quê:** manter as cópias daria peso maior às músicas populares, que
  costumam estar em mais gêneros, e colocaria a mesma faixa no treino e no teste.

## D2 · Mesma música com `track_id` diferentes (preparar.py / treinar.py)

- **Fato:** 4.747 músicas (mesmo artista + nome) têm mais de um `track_id`
  (versão do álbum, single, coletânea, remaster...), com até 45 versões.
- **Decisão:** manter as versões (a popularidade delas é diferente de verdade),
  mas criar `musica_id` = artista + nome normalizados e usá-lo como **grupo** na
  divisão treino/teste e na validação cruzada (`GroupShuffleSplit` / `GroupKFold`).
- **Por quê:** sem isso, o modelo "decora" a música no treino e acerta a outra
  versão no teste, inflando as métricas (vazamento de dados).

## D3 · Popularidade zero (preparar.py / treinar.py)

- **Fato:** no CSV original, 16.020 linhas (14,1%) têm `popularity = 0`; após D1
  restam 9.347 faixas (10,4%). O histograma mostra um pico isolado em 0, fora da
  forma do resto da distribuição (`resultados/figuras/01_distribuicao_popularidade.png`).
- **Evidências de que o zero é artefato da coleta, e não "música impopular":**
  - 2.881 dessas faixas têm outra versão da mesma música com popularidade > 0;
  - 6.185 (66%) são de artistas que têm outras faixas com popularidade > 0;
  - os zeros se concentram em poucos gêneros (jazz 74%, iranian 66%, romance
    61%, latin 61%, country 59%), enquanto 19 gêneros quase não têm zeros
    (`resultados/figuras/03_zeros_por_genero.png`);
  - os atributos de áudio médios das faixas com zero são parecidos com os das
    demais (`resultados/eda.json`, chave `media_atributos_zero_vs_demais`).
- **Decisão:** tratar `popularity = 0` como **alvo ausente** e treinar o modelo
  principal **sem essas faixas**. Como experimento de comparação, treinar
  também com os zeros (`python treinar.py --com-zeros`) e reportar ambos.

## D4 · Valores inválidos (preparar.py)

- `tempo = 0` (157 faixas) e `time_signature` 0 ou 1 (1.008 faixas) são
  estimativas inválidas da API → convertidos em ausentes e preenchidos dentro do
  pipeline (`SimpleImputer`: mediana / mais frequente), aprendendo os valores
  só no treino.

## D5 · Atributos

- Removidos: `Unnamed: 0`, `track_id`, `artists`, `album_name`, `track_name`
  (identificadores; o artista seria um atalho que não existe para uma música nova).
- `duration_ms` → `duracao_min` (mais fácil de ler na interface e na explicação).
- `explicit` → 0/1; `key`, `time_signature` e `track_genre` → one-hot.

## Nota para a discussão (artigo) · R² esperado é baixo

- Nenhum atributo de áudio tem correlação de Spearman com a popularidade acima
  de |0,20| (o maior é `instrumentalness`, −0,197).
- Só o gênero explica **R² ≈ 0,44** (prever a média do gênero), ou seja, a maior
  parte do que dá para prever vem do gênero.
- A popularidade depende de fatores que **não estão no dataset**: artista,
  gravadora, marketing, playlists editoriais, data de lançamento, viralização.
- Portanto, um R² moderado não é falha de modelagem: é um limite do problema
  com estes dados, e deve ser discutido como tal.
