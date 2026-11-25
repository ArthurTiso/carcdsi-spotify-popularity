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
- **Revisão (grupo_id):** uma auditoria depois do primeiro treino achou 119
  faixas do teste (0,7%) com **áudio idêntico** a faixas do treino, publicadas
  com outro título — "Left and Right (Feat. Jung Kook of BTS)" × "Left and
  Right", "Der Fluss" × "El río" — que `musica_id` não unia. Agora o grupo
  (`grupo_id`) une, por *union-find*, faixas com o mesmo `musica_id` **ou** os
  mesmos atributos de áudio: 81.206 músicas → 80.810 grupos (366 uniões; o maior
  grupo continua com 45 faixas da mesma música). Os pares encontrados também
  mostram ruído no alvo: a mesma gravação aparece com popularidade 75 e 1.

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

## D6 · Seleção e avaliação dos modelos (treinar.py)

- Modelos comparados: os quatro da Atividade 2 (DummyRegressor, Ridge,
  Random Forest, HistGradientBoosting) + uma referência extra, **média do
  gênero**, para medir quanto os atributos de áudio acrescentam ao gênero.
- Hiperparâmetros ajustados com `GridSearchCV` (Ridge) e `RandomizedSearchCV`
  (Random Forest: 8 combinações; HistGradientBoosting: 15) usando
  **GroupKFold com 5 folds só no treino** e RMSE como critério.
- O melhor modelo é escolhido pelo **RMSE da validação cruzada**, não pelo teste;
  o teste (20%, nunca visto) só é usado para relatar MAE, MSE, RMSE e R².
- As previsões são limitadas a 0..100 antes de calcular as métricas.
- Importância dos atributos por **permutação** no teste (6.000 faixas, 5
  repetições), medida como aumento do RMSE; o gênero é permutado como um único
  atributo (e não coluna one-hot por coluna).
- Experimentos extras com os mesmos hiperparâmetros: (a) incluindo a
  popularidade zero; (b) com divisão aleatória em vez de agrupada, para medir o
  efeito do vazamento entre versões da mesma música.

## D7 · Explicação e LLM (prever.py / processar.py / app.py)

- **SHAP** (`TreeExplainer`) explica cada previsão; as colunas one-hot são
  somadas de volta ao atributo original, e o programa confere que
  valor base + contribuições = previsão antes de gravar o resultado.
- A **LLM não calcula nada**: ela recebe o JSON produzido pelo modelo (relato e
  chat) ou pede ao programa que execute o modelo por *tool calling*
  (assistente). Isso segue o padrão das aulas: o programa executa e devolve; a
  LLM interpreta.
- No assistente, só o gênero é obrigatório; os atributos que a descrição não
  permite estimar recebem o **valor típico (mediana) do gênero no treino**, e o
  resultado informa quais campos foram estimados e quais foram preenchidos.
- O gênero é restrito por `enum` na definição da ferramenta (114 gêneros) e a
  ficha passa pela mesma validação explícita de `prever.py`.
- Modelo de LLM padrão: `nvidia/nemotron-3.5-bolt-e4b` (o `openai/gpt-oss-20b`
  usado nas aulas foi retirado do catálogo da NVIDIA). Nos testes, o
  `nemotron-3.5-lightning-30b-a3b` também funcionou, mas foi ~2,5× mais lento.
- O intervalo exibido é **previsão ± MAE do teste**; os prompts deixam claro
  que não é a faixa de popularidade do gênero (a LLM confundiu os dois nos
  primeiros testes).
- **Fatos prontos em vez de dedução.** Nos testes, a LLM errou coisas que o
  programa sabe calcular: disse que 47,4 era "inferior" a 46,7, que a
  tonalidade 5 era Ré (é Fá) e que −6,9 dB era "volume baixo" (é alto para o
  dataset). Correção: `prever.py` grava frases prontas de comparação
  (`"4.2 pontos acima da média (60.6)"`), nomes legíveis de tonalidade, modo e
  compasso, e as descrições dos atributos trazem a mediana do dataset. Regra
  geral adotada: **o programa calcula; a LLM só redige**.
- **Números arredondados e temperatura 0.** O JSON enviado à LLM leva os números
  com 1 casa (atributos de áudio com 3), e as chamadas usam temperatura 0.
- **Limpeza de LaTeX no código.** Mesmo instruído, o modelo escrevia `$\pm$`
  em boa parte das respostas; `limpar_texto()` converte para texto simples.

## Nota para a discussão (artigo) · R² esperado é baixo

- Nenhum atributo de áudio tem correlação de Spearman com a popularidade acima
  de |0,20| (o maior é `instrumentalness`, −0,197).
- Só o gênero explica **R² ≈ 0,44** no dataset todo (0,42 no teste) ao prever a média do gênero; a maior
  parte do que dá para prever vem do gênero.
- A popularidade depende de fatores que **não estão no dataset**: artista,
  gravadora, marketing, playlists editoriais, data de lançamento, viralização.
- Portanto, um R² moderado não é falha de modelagem: é um limite do problema
  com estes dados, e deve ser discutido como tal.

## Resultados finais (treinar.py, execução completa)

Teste: 16.066 faixas; treino: 64.327. Tempo total: ~19 min em 4 núcleos.

| Modelo | RMSE CV | MAE | MSE | RMSE | R² | Hiperparâmetros escolhidos |
|---|---:|---:|---:|---:|---:|---|
| Dummy (média) | 18,16 | 15,09 | 327,5 | 18,10 | 0,000 | — |
| Média do gênero | 13,59 | 9,85 | 188,9 | 13,75 | 0,423 | — |
| Ridge | 13,52 | 9,79 | 187,0 | 13,68 | 0,429 | α = 1 |
| Random Forest | 13,99 | 10,42 | 197,9 | 14,07 | 0,396 | max_depth 30, min_samples_leaf 10, max_features 0,33 |
| **HistGradientBoosting** | **13,17** | **9,31** | **176,9** | **13,30** | **0,460** | learning_rate 0,03, max_iter 600, max_leaf_nodes 63, min_samples_leaf 20, l2 1,0 |

Leituras para a discussão:

1. **Ranking igual na validação cruzada e no teste.** A escolha pelo RMSE da
   validação cruzada coincide com o melhor no teste, sinal de que a avaliação
   é estável.
2. **Random Forest ficou abaixo do Ridge.** Com 114 colunas one-hot de gênero,
   a floresta sorteia poucas colunas de gênero por divisão e aprende pior o
   efeito do gênero; o boosting, que corrige erros em sequência, aproveita melhor.
3. **Ganho do áudio sobre o gênero é pequeno:** R² 0,423 → 0,460 (+0,037);
   MAE 9,85 → 9,31.
4. **Importância por permutação** (aumento do RMSE, pontos): gênero 6,82;
   instrumentalness 0,32; energy 0,23; loudness 0,18; duração 0,16; demais
   < 0,10; key, mode e time_signature ≈ 0.
5. **Viés por faixa de popularidade real** (previsto − real): 1–20: +11,4;
   21–40: +4,7; 41–60: −6,0; 61–80: −16,9; 81–100: −32,6. O modelo encolhe as
   previsões em direção à média; não identifica hits.
6. **Experimentos:**
   - com popularidade zero: R² 0,354, MAE 11,72 (conjunto de teste diferente,
     com zeros; o modelo tenta prever um valor que é artefato da coleta);
   - divisão aleatória (sem agrupar): R² 0,477, MAE 9,18, com 1.216 faixas do
     teste tendo outra versão da mesma música no treino. O ganho (+0,017) é
     otimismo causado pelo vazamento, não melhora real.

## Avaliação da LLM (avaliar_llm.py)

`avaliar_llm.py` gera relatos com a LLM (`nvidia/nemotron-3.5-bolt-e4b`) e
confere automaticamente se cada número citado existe no JSON enviado a ela
(aceitando arredondamento e as diferenças previsão − médias). Rodada final:
30 relatos por configuração, alternando cinco músicas (pop da `ficha.json` e
fichas típicas de samba, metal, classical e k-pop) — `resultados/avaliacao_llm.json`.

| Configuração | Número fora do JSON | Resposta com LaTeX | Tempo médio |
|---|---:|---:|---:|
| prompt original (JSON completo, temperatura padrão, sem limpeza) | 0/30 | 0/30 | 3,0 s |
| prompt revisado (JSON completo, temperatura padrão, sem limpeza) | 0/30 | 6/30 | 3,5 s |
| **final** (prompt revisado + números arredondados + temperatura 0 + limpeza) | **0/30** | **0/30** | 3,3 s |

Leitura honesta dos resultados:

- Nesta amostra, **nenhuma configuração citou número inexistente**. Os erros de
  cópia vistos nos testes iniciais (ex.: "65,8" para uma previsão de 64,8, em
  3 de 8 respostas) foram **intermitentes**; as salvaguardas (arredondar,
  temperatura 0, instrução explícita) são baratas e foram mantidas, mas esta
  avaliação não mostra diferença estatística entre as configurações nesse item.
- O **LaTeX** aparece de forma irregular sem a limpeza (0 a 6 de 30 por rodada;
  6 de 12 numa rodada preliminar) e **some com `limpar_texto()`**.
- O verificador automático só confere números. Os **erros de raciocínio**
  ("47,4 é inferior a 46,7", "tonalidade 5 = Ré") foram tratados por
  construção — o programa entrega as comparações e os nomes prontos — e
  conferidos manualmente no terminal e na interface.
- Limitação: uma única LLM e um único tipo de pergunta (o relato inicial);
  as respostas do chat e do assistente foram verificadas só manualmente.
