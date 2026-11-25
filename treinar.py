"""Etapa 3 · Treinar, comparar e salvar o modelo de regressão de popularidade.

Fluxo:
  1. Lê dados/processado/spotify_limpo.csv e remove a popularidade zero
     (decisão D3 em docs/decisoes.md; use --com-zeros para incluí-la).
  2. Divide 80/20 com GroupShuffleSplit: versões da mesma música (musica_id)
     ficam sempre no mesmo conjunto (decisão D2).
  3. Ajusta os hiperparâmetros com validação cruzada GroupKFold (5 folds)
     apenas no treino, escolhe o melhor modelo pelo RMSE da validação cruzada
     e só então mede todos no teste (MAE, MSE, RMSE, R²).
  4. Calcula a importância por permutação do melhor modelo no teste.
  5. Roda dois experimentos de comparação com os hiperparâmetros escolhidos:
     (a) incluindo a popularidade zero; (b) com divisão aleatória (sem agrupar).
  6. Salva o pipeline completo (preparação + modelo) em modelos/modelo.joblib
     e as métricas/figuras em resultados/.

Execução:
  python treinar.py                # completo (~15-25 min em 4 núcleos)
  python treinar.py --rapido       # busca menor, para testar o código
  python treinar.py --com-zeros    # modelo principal treinado COM os zeros
"""
import argparse
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score)
from sklearn.model_selection import (GridSearchCV, GroupKFold,
                                     GroupShuffleSplit, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (ALVO, ARQUIVO_MODELO, ARQUIVO_ORIGEM, ATRIBUTOS, BINARIAS,
                    CATEGORICAS, DADOS_LIMPOS, DESCRICOES, GRUPO, NUMERICAS,
                    PASTA_FIGURAS, PASTA_RESULTADOS, SEMENTE)
from explorar import AZUL, LARANJA, TEXTO_2, salvar  # mesmo estilo de figura

warnings.filterwarnings("ignore", category=UserWarning)

N_FOLDS = 5
METRICA_CV = "neg_root_mean_squared_error"


# ---------------------------------------------------------------- preparação
def criar_preparo(padronizar):
    """ColumnTransformer: imputação (+ padronização) e one-hot.

    As estatísticas de preenchimento/padronização são aprendidas apenas no
    fit do treino (dentro de cada fold na validação cruzada).
    """
    numericas = [("preencher", SimpleImputer(strategy="median"))]
    if padronizar:
        numericas.append(("padronizar", StandardScaler()))
    categorias = Pipeline([
        ("preencher", SimpleImputer(strategy="most_frequent")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numericas", Pipeline(numericas), NUMERICAS),
        ("binarias", "passthrough", BINARIAS),
        ("categorias", categorias, CATEGORICAS),
    ])


def candidatos(rapido):
    """Modelos da Atividade 2 + referências, com o espaço de busca de cada um."""
    n_rf, n_hgb = (2, 3) if rapido else (8, 15)
    return {
        "Dummy (média)": {
            "pipeline": Pipeline([("preparo", criar_preparo(False)),
                                  ("modelo", DummyRegressor(strategy="mean"))]),
            "busca": None,
        },
        "Média do gênero": {
            # Referência extra: Ridge quase sem penalidade só com o gênero
            # = prever a média de popularidade do gênero.
            "pipeline": Pipeline([
                ("preparo", ColumnTransformer([("genero", OneHotEncoder(
                    handle_unknown="ignore"), ["track_genre"])])),
                ("modelo", Ridge(alpha=1e-6))]),
            "busca": None,
        },
        "Ridge": {
            "pipeline": Pipeline([("preparo", criar_preparo(True)),
                                  ("modelo", Ridge())]),
            "busca": ("grade", {"modelo__alpha": [0.1, 1, 10, 100, 1000]}),
        },
        "Random Forest": {
            "pipeline": Pipeline([("preparo", criar_preparo(False)),
                                  ("modelo", RandomForestRegressor(
                                      n_estimators=150, max_samples=0.5,
                                      n_jobs=-1, random_state=SEMENTE))]),
            "busca": ("aleatoria", n_rf, {
                "modelo__max_depth": [None, 20, 30],
                "modelo__min_samples_leaf": [2, 5, 10, 20],
                "modelo__max_features": [0.2, 0.33, 0.5],
            }),
        },
        "HistGradientBoosting": {
            "pipeline": Pipeline([("preparo", criar_preparo(False)),
                                  ("modelo", HistGradientBoostingRegressor(
                                      early_stopping=False,
                                      random_state=SEMENTE))]),
            "busca": ("aleatoria", n_hgb, {
                "modelo__learning_rate": [0.03, 0.05, 0.1],
                "modelo__max_iter": [300, 600, 1000],
                "modelo__max_leaf_nodes": [15, 31, 63, 127],
                "modelo__min_samples_leaf": [20, 50, 100],
                "modelo__l2_regularization": [0.0, 1.0, 10.0],
            }),
        },
    }


# ------------------------------------------------------------------ métricas
def metricas(y_real, y_previsto):
    y_previsto = np.clip(y_previsto, 0, 100)  # popularidade só existe em 0..100
    mse = mean_squared_error(y_real, y_previsto)
    return {"MAE": round(float(mean_absolute_error(y_real, y_previsto)), 3),
            "MSE": round(float(mse), 3),
            "RMSE": round(float(np.sqrt(mse)), 3),
            "R2": round(float(r2_score(y_real, y_previsto)), 4)}


def ajustar(nome, item, X, y, grupos, cv):
    """Ajusta o modelo; com busca, escolhe hiperparâmetros por CV agrupada."""
    inicio = time.time()
    busca = item["busca"]
    if busca is None:
        modelo = item["pipeline"]
        cv_rmse = -np.mean(cross_val_score(
            modelo, X, y, groups=grupos, cv=cv, scoring=METRICA_CV))
        modelo.fit(X, y)
        melhores = {}
    else:
        if busca[0] == "grade":
            procura = GridSearchCV(item["pipeline"], busca[1], cv=cv,
                                   scoring=METRICA_CV)
        else:
            procura = RandomizedSearchCV(item["pipeline"], busca[2],
                                         n_iter=busca[1], cv=cv,
                                         scoring=METRICA_CV,
                                         random_state=SEMENTE)
        procura.fit(X, y, groups=grupos)
        modelo, cv_rmse = procura.best_estimator_, -procura.best_score_
        melhores = {k.replace("modelo__", ""): v
                    for k, v in procura.best_params_.items()}
    print(f"  {nome:<22} RMSE CV = {cv_rmse:6.3f}   {melhores}   "
          f"({time.time() - inicio:.0f}s)")
    return modelo, round(float(cv_rmse), 3), melhores


# ------------------------------------------------------------------- figuras
def figura_comparacao(tabela):
    nomes = list(tabela)
    fig, eixos = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, chave, titulo in [(eixos[0], "RMSE", "RMSE no teste (menor é melhor)"),
                              (eixos[1], "R2", "R² no teste (maior é melhor)")]:
        # abs() evita exibir "-0.000" quando o valor arredondado é zero
        valores = [v if abs(v) >= 5e-4 else 0.0
                   for v in (tabela[n]["teste"][chave] for n in nomes)]
        melhor = (np.argmin if chave == "RMSE" else np.argmax)(valores)
        cores = [LARANJA if i == melhor else AZUL for i in range(len(nomes))]
        ax.barh(nomes, valores, color=cores, height=0.65)
        for i, v in enumerate(valores):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7.5, color=TEXTO_2)
        ax.set_title(titulo, fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, max(valores) * 1.25)
    eixos[0].invert_yaxis()
    salvar(fig, "06_comparacao_modelos.png")


def figura_real_vs_previsto(y_real, y_previsto, nome):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    hb = ax.hexbin(y_real, np.clip(y_previsto, 0, 100), gridsize=45, cmap="Blues",
                   mincnt=1, bins="log", linewidths=0)
    ax.plot([0, 100], [0, 100], color=LARANJA, linewidth=1.2, label="previsão perfeita")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="popularidade real",
           ylabel="popularidade prevista",
           title=f"Real × previsto no teste · {nome}")
    ax.legend(frameon=False, loc="upper left")
    fig.colorbar(hb, ax=ax, label="faixas (escala log)", shrink=0.8)
    salvar(fig, "07_real_vs_previsto.png")


def erro_por_faixa(y_real, y_previsto):
    """MAE e viés (previsto - real) por faixa de popularidade real."""
    previsto = np.clip(y_previsto, 0, 100)
    faixas = pd.cut(y_real, [0, 20, 40, 60, 80, 100],
                    labels=["1-20", "21-40", "41-60", "61-80", "81-100"])
    tabela = pd.DataFrame({"faixa": faixas, "erro": previsto - y_real})
    resumo = tabela.groupby("faixa", observed=False)["erro"].agg(
        faixas="size", MAE=lambda e: e.abs().mean(), vies="mean")
    return resumo.round(2)


def figura_erro_por_faixa(resumo, nome):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    posicoes = np.arange(len(resumo))
    cores = [LARANJA if v < 0 else AZUL for v in resumo["vies"]]
    ax.bar(posicoes, resumo["vies"], color=cores, width=0.6)
    for i, (v, n) in enumerate(zip(resumo["vies"], resumo["faixas"])):
        ax.text(i, v + (0.6 if v >= 0 else -0.6), f"{v:+.1f}\n(n={n:,})".replace(",", "."),
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5,
                color=TEXTO_2)
    ax.axhline(0, color=TEXTO_2, linewidth=0.8)
    ax.set_xticks(posicoes, resumo.index)
    limite = resumo["vies"].abs().max() * 1.6
    ax.set(ylim=(-limite, limite), xlabel="popularidade real",
           ylabel="viés médio (previsto − real)",
           title=f"Viés por faixa de popularidade · {nome}")
    ax.grid(axis="x", visible=False)
    salvar(fig, "09_vies_por_faixa.png")


def figura_importancia(importancia, nome):
    serie = importancia.sort_values("media")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(serie.index, serie["media"], xerr=serie["desvio"], color=AZUL,
            height=0.65, error_kw={"ecolor": TEXTO_2, "elinewidth": 0.8})
    for i, v in enumerate(serie["media"]):
        ax.text(v + serie["desvio"].iloc[i], i, f"  {v:.2f}", va="center",
                fontsize=7, color=TEXTO_2)
    ax.set(title=f"Importância por permutação · {nome}",
           xlabel="aumento do RMSE no teste ao embaralhar o atributo (pontos)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    salvar(fig, "08_importancia_permutacao.png")


# --------------------------------------------------------------- experimentos
def experimento(dados, modelo_base, agrupar):
    """Retreina o melhor pipeline (mesmos hiperparâmetros) em outro cenário."""
    X, y = dados[ATRIBUTOS], dados[ALVO].to_numpy()
    if agrupar:
        treino, teste = next(GroupShuffleSplit(
            n_splits=1, test_size=0.2, random_state=SEMENTE).split(
                X, y, dados[GRUPO]))
    else:
        treino, teste = train_test_split(np.arange(len(dados)), test_size=0.2,
                                         random_state=SEMENTE)
    modelo = clone(modelo_base).fit(X.iloc[treino], y[treino])
    musicas_treino = set(dados[GRUPO].iloc[treino])
    vazadas = int(dados[GRUPO].iloc[teste].isin(musicas_treino).sum())
    return {**metricas(y[teste], modelo.predict(X.iloc[teste])),
            "n_treino": len(treino), "n_teste": len(teste),
            "faixas_teste_com_mesma_musica_no_treino": vazadas}


# ---------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rapido", action="store_true",
                        help="busca de hiperparâmetros reduzida (teste do código)")
    parser.add_argument("--com-zeros", action="store_true",
                        help="mantém as faixas com popularidade zero no modelo principal")
    args = parser.parse_args()
    inicio = time.time()

    todos = pd.read_csv(DADOS_LIMPOS)
    dados = todos if args.com_zeros else todos[todos["popularidade_zero"] == 0]
    dados = dados.reset_index(drop=True)
    X, y, grupos = dados[ATRIBUTOS], dados[ALVO].to_numpy(), dados[GRUPO]

    divisor = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMENTE)
    treino, teste = next(divisor.split(X, y, grupos))
    Xt, Xe, yt, ye, gt = X.iloc[treino], X.iloc[teste], y[treino], y[teste], grupos.iloc[treino]
    assert not set(gt) & set(grupos.iloc[teste]), "música no treino e no teste"
    print(f"Faixas: {len(dados)} | treino: {len(Xt)} | teste: {len(Xe)} | "
          f"zeros {'incluídos' if args.com_zeros else 'removidos'}")

    cv = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMENTE)
    print(f"\nAjuste com validação cruzada agrupada ({N_FOLDS} folds) no treino:")
    tabela, modelos = {}, {}
    for nome, item in candidatos(args.rapido).items():
        modelo, cv_rmse, melhores = ajustar(nome, item, Xt, yt, gt, cv)
        modelos[nome] = modelo
        tabela[nome] = {"cv_rmse": cv_rmse, "hiperparametros": melhores,
                        "teste": metricas(ye, modelo.predict(Xe))}

    # Escolha pelo erro da validação cruzada (o teste fica só para relatar).
    nome_melhor = min(tabela, key=lambda n: tabela[n]["cv_rmse"])
    melhor = modelos[nome_melhor]
    print(f"\nMelhor modelo (menor RMSE na validação cruzada): {nome_melhor}")
    print("\nMétricas no TESTE:")
    print(pd.DataFrame({n: t["teste"] for n, t in tabela.items()}).T.to_string())

    previsto = melhor.predict(Xe)
    figura_comparacao(tabela)
    figura_real_vs_previsto(ye, previsto, nome_melhor)
    erro_faixas = erro_por_faixa(ye, previsto)
    figura_erro_por_faixa(erro_faixas, nome_melhor)
    print("\nErro do melhor modelo por faixa de popularidade real:")
    print(erro_faixas.to_string())

    print("\nImportância por permutação no teste (pode levar alguns minutos)...")
    amostra = np.random.default_rng(SEMENTE).choice(len(Xe), size=min(6000, len(Xe)),
                                                    replace=False)
    perm = permutation_importance(melhor, Xe.iloc[amostra], ye[amostra],
                                  scoring=METRICA_CV, n_repeats=5,
                                  random_state=SEMENTE, n_jobs=1)
    importancia = pd.DataFrame({"media": perm.importances_mean,
                                "desvio": perm.importances_std}, index=ATRIBUTOS)
    figura_importancia(importancia, nome_melhor)
    print(importancia.sort_values("media", ascending=False).round(3).to_string())

    print("\nExperimentos com os hiperparâmetros escolhidos:")
    experimentos = {
        "principal (agrupada, sem zeros)" if not args.com_zeros
        else "principal (agrupada, com zeros)": {**tabela[nome_melhor]["teste"],
                                                  "faixas_teste_com_mesma_musica_no_treino": 0},
        "com zeros (agrupada)": experimento(todos, melhor, agrupar=True),
        "divisao aleatoria (sem zeros)": experimento(
            todos[todos["popularidade_zero"] == 0].reset_index(drop=True),
            melhor, agrupar=False),
    }
    print(pd.DataFrame(experimentos).T.to_string())

    # Faixas válidas para a interface/validação da ficha (vindas do treino).
    intervalos = {c: {"min": float(Xt[c].min()), "max": float(Xt[c].max()),
                      "mediana": float(Xt[c].median())} for c in NUMERICAS}
    categorias = {c: sorted(int(v) if c != "track_genre" else v
                            for v in Xt[c].dropna().unique()) for c in CATEGORICAS}
    # Contexto descritivo para a explicação (calculado só no treino).
    por_genero = pd.DataFrame({"genero": Xt["track_genre"], "pop": yt}) \
        .groupby("genero")["pop"].agg(["mean", "count"])
    # Valores típicos (mediana / mais frequente) de cada atributo por gênero:
    # preenchem campos não informados na interface e na ferramenta da LLM.
    tipicos = {}
    for genero, grupo in Xt.groupby("track_genre"):
        tipicos[genero] = {c: round(float(grupo[c].median()), 3) for c in NUMERICAS}
        for c in BINARIAS + ["key", "time_signature"]:
            tipicos[genero][c] = int(grupo[c].mode().iloc[0])
    contexto_treino = {
        "valores_tipicos_por_genero": tipicos,
        "popularidade_media": round(float(yt.mean()), 2),
        "popularidade_desvio": round(float(yt.std()), 2),
        "popularidade_media_por_genero": {
            g: {"media": round(float(r["mean"]), 2), "faixas": int(r["count"])}
            for g, r in por_genero.iterrows()},
    }
    avaliacao = {
        "modelo": nome_melhor,
        "teste": tabela[nome_melhor]["teste"],
        "referencia_dummy": tabela["Dummy (média)"]["teste"],
        "referencia_media_genero": tabela["Média do gênero"]["teste"],
        "n_treino": len(Xt), "n_teste": len(Xe),
        "divisao": "80/20 GroupShuffleSplit por musica_id, random_state=42",
        "validacao_cruzada": f"GroupKFold {N_FOLDS} folds no treino, métrica RMSE",
        "popularidade_zero": "incluída" if args.com_zeros else "removida (alvo ausente)",
    }
    versoes = {"pandas": pd.__version__, "sklearn": sklearn.__version__}
    origem = json.loads(ARQUIVO_ORIGEM.read_text(encoding="utf-8"))
    artefato = {"modelo": melhor, "nome": nome_melhor, "avaliacao": avaliacao,
                "importancia_permutacao": importancia["media"].round(4).to_dict(),
                "atributos": {"numericas": NUMERICAS, "binarias": BINARIAS,
                              "categoricas": CATEGORICAS},
                "intervalos": intervalos, "categorias": categorias,
                "contexto_treino": contexto_treino,
                "descricoes": DESCRICOES, "origem": origem, "versoes": versoes}

    if args.com_zeros:
        sufixo = "_com_zeros"
        print("\n--com-zeros: o modelo NÃO substitui modelos/modelo.joblib.")
    else:
        sufixo = ""
        ARQUIVO_MODELO.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artefato, ARQUIVO_MODELO)
        print("\nMODELO SALVO:", ARQUIVO_MODELO)

    relatorio = {"avaliacao": avaliacao, "modelos": tabela,
                 "importancia_permutacao": importancia.round(4).to_dict(orient="index"),
                 "experimentos": experimentos,
                 "erro_por_faixa_real": erro_faixas.to_dict(orient="index"),
                 "versoes": versoes,
                 "busca_rapida": args.rapido,
                 "tempo_total_s": round(time.time() - inicio)}
    caminho = PASTA_RESULTADOS / f"metricas{sufixo}.json"
    caminho.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print("MÉTRICAS:", caminho)
    print(json.dumps(avaliacao, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
