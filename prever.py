"""Etapa 4 · Ler uma ficha de música, prever a popularidade e explicar com SHAP.

Carrega o pipeline salvo por treinar.py (modelos/modelo.joblib), lê a ficha
(ficha.json por padrão), valida cada campo explicitamente, calcula a
popularidade prevista e as contribuições SHAP de cada atributo e grava tudo
em resultado.json, que será lido pela LLM em processar.py.

As funções carregar_artefato(), validar_ficha() e prever() também são usadas
pela interface (app.py) e pela ferramenta chamada pela LLM (processar.py).

Execução:
  python prever.py                       # usa ficha.json
  python prever.py --ficha outra.json
"""
import argparse
import json
import math
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn

from config import (ARQUIVO_MODELO, ATRIBUTOS, BINARIAS, NUMERICAS, PASTA,
                    PASTA_RESULTADOS)

# Limites de domínio de cada atributo numérico (definição do Spotify).
LIMITES = {
    "danceability": (0, 1), "energy": (0, 1), "speechiness": (0, 1),
    "acousticness": (0, 1), "instrumentalness": (0, 1), "liveness": (0, 1),
    "valence": (0, 1), "loudness": (-60, 5), "tempo": (30, 250),
    "duracao_min": (0.1, 60),
}

LIMITES_INTERPRETACAO = [
    "a previsão é uma estimativa do modelo, não uma garantia de sucesso",
    "SHAP descreve como ESTE modelo chegou ao número; não prova causa",
    "popularidade depende de fatores ausentes nos dados: artista, marketing, "
    "playlists, data de lançamento, viralização",
    "o modelo foi treinado com faixas do Spotify coletadas em 2022 (dataset do Kaggle)",
]


def carregar_artefato(caminho=ARQUIVO_MODELO):
    """Carrega somente um artefato criado por treinar.py, com as mesmas versões."""
    if not caminho.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute treinar.py primeiro.")
    artefato = joblib.load(caminho)
    if artefato["versoes"] != {"pandas": pd.__version__, "sklearn": sklearn.__version__}:
        raise ValueError("Use as mesmas versões de pandas/scikit-learn do treino "
                         f"({artefato['versoes']}).")
    return artefato


def validar_ficha(ficha, artefato):
    """Validação explícita, campo a campo. Retorna a ficha normalizada."""
    if not isinstance(ficha, dict) or set(ficha) != set(ATRIBUTOS):
        faltam = sorted(set(ATRIBUTOS) - set(ficha or {}))
        sobram = sorted(set(ficha or {}) - set(ATRIBUTOS))
        raise ValueError(f"A ficha precisa ter exatamente os campos {ATRIBUTOS}. "
                         f"Faltam: {faltam}. Sobram: {sobram}.")
    limpa = {}
    for campo in NUMERICAS:
        valor = ficha[campo]
        if isinstance(valor, bool) or not isinstance(valor, (int, float)) \
                or not math.isfinite(valor):
            raise ValueError(f"{campo}: informe um número finito.")
        minimo, maximo = LIMITES[campo]
        if not minimo <= valor <= maximo:
            raise ValueError(f"{campo}: deve estar entre {minimo} e {maximo}.")
        limpa[campo] = float(valor)
    for campo in BINARIAS:
        if type(ficha[campo]) is not int or ficha[campo] not in (0, 1):
            raise ValueError(f"{campo}: use 0 ou 1.")
        limpa[campo] = int(ficha[campo])
    categorias = artefato["categorias"]
    for campo in ["key", "time_signature"]:
        valor = ficha[campo]
        if isinstance(valor, bool) or not isinstance(valor, int) \
                or valor not in categorias[campo]:
            raise ValueError(f"{campo}: use um destes valores {categorias[campo]}.")
        limpa[campo] = valor
    genero = ficha["track_genre"]
    if not isinstance(genero, str) or genero.strip().lower() not in categorias["track_genre"]:
        raise ValueError(f"track_genre '{genero}' não existe no treino. "
                         "Veja a lista de gêneros em artefato['categorias'].")
    limpa["track_genre"] = genero.strip().lower()
    return limpa


def completar_ficha(parcial, artefato):
    """Completa uma ficha parcial com os valores típicos do gênero (treino).

    Usada pela interface e pela ferramenta da LLM, quando a pessoa descreve
    só parte da música. Retorna (ficha_completa, campos_preenchidos).
    """
    if not isinstance(parcial, dict) or "track_genre" not in parcial:
        raise ValueError("Informe pelo menos o gênero (track_genre).")
    desconhecidos = sorted(set(parcial) - set(ATRIBUTOS))
    if desconhecidos:
        raise ValueError(f"Campos desconhecidos: {desconhecidos}.")
    genero = str(parcial["track_genre"]).strip().lower()
    tipicos = artefato["contexto_treino"]["valores_tipicos_por_genero"]
    if genero not in tipicos:
        raise ValueError(f"track_genre '{genero}' não existe no treino.")
    ficha, preenchidos = {"track_genre": genero}, []
    for campo in ATRIBUTOS:
        if campo == "track_genre":
            continue
        valor = parcial.get(campo)
        if valor is None:
            valor = tipicos[genero][campo]
            preenchidos.append(campo)
        elif campo not in NUMERICAS and isinstance(valor, float) and valor.is_integer():
            valor = int(valor)  # LLMs às vezes mandam 1.0 em vez de 1
        ficha[campo] = valor
    return ficha, preenchidos


def contribuicoes_por_atributo(preparo, valores):
    """Soma as colunas one-hot de volta ao atributo original."""
    contrib = {}
    for nome, fatia in preparo.output_indices_.items():
        if nome == "remainder" or fatia.stop == fatia.start:
            continue
        colunas = dict(preparo.named_transformers_.items())[nome]
        entradas = [t[2] for t in preparo.transformers_ if t[0] == nome][0]
        if nome == "categorias":
            codificador = colunas.named_steps["codificar"]
            inicio = fatia.start
            for atributo, cats in zip(entradas, codificador.categories_):
                contrib[atributo] = float(valores[inicio:inicio + len(cats)].sum())
                inicio += len(cats)
        else:
            for atributo, v in zip(entradas, valores[fatia]):
                contrib[atributo] = float(v)
    return contrib


def prever(ficha, artefato):
    """Previsão + explicação SHAP de uma ficha já validada."""
    modelo = artefato["modelo"]
    preparo, estimador = modelo.named_steps["preparo"], modelo.named_steps["modelo"]
    entrada = pd.DataFrame([ficha])[ATRIBUTOS]
    bruto = float(modelo.predict(entrada)[0])
    popularidade = float(np.clip(bruto, 0, 100))

    # SHAP recebe o estimador e a mesma representação usada para prever.
    transformada = preparo.transform(entrada)
    explicador = shap.TreeExplainer(estimador)
    explicacao = explicador(transformada)
    valores = np.asarray(explicacao.values[0], dtype=float)
    base = float(np.ravel(explicacao.base_values)[0])
    if not np.isclose(base + valores.sum(), bruto, atol=1e-3):
        raise ValueError("A explicação SHAP não reconstrói a previsão.")
    contrib = contribuicoes_por_atributo(preparo, valores)
    contrib = dict(sorted(contrib.items(), key=lambda kv: -abs(kv[1])))

    contexto = artefato["contexto_treino"]
    generos = contexto["popularidade_media_por_genero"]
    ranking = sorted(generos, key=lambda g: -generos[g]["media"])
    genero = ficha["track_genre"]
    avaliacao = artefato["avaliacao"]
    return {
        "versao": 1,
        "atributos": ficha,
        "descricao_atributos": artefato["descricoes"],
        "previsao": {
            "popularidade_prevista": round(popularidade, 2),
            "escala": "0 a 100 (índice de popularidade do Spotify)",
            "erro_medio_absoluto_no_teste": avaliacao["teste"]["MAE"],
            "intervalo_da_previsao": [
                round(max(0.0, popularidade - avaliacao["teste"]["MAE"]), 1),
                round(min(100.0, popularidade + avaliacao["teste"]["MAE"]), 1)],
            "intervalo_da_previsao_significa": "previsão ± erro médio absoluto do "
                "modelo no teste (não é a faixa de popularidade do gênero)",
            "modelo": artefato["nome"],
        },
        "contexto_treino": {
            "popularidade_media_geral": contexto["popularidade_media"],
            "popularidade_media_do_genero": generos[genero]["media"],
            "faixas_do_genero_no_treino": generos[genero]["faixas"],
            "posicao_do_genero": f"{ranking.index(genero) + 1}º de {len(ranking)} gêneros",
        },
        "explicacao_shap": {
            "saida_explicada": "popularidade prevista (antes de limitar a 0..100)",
            "metodo": "shap.TreeExplainer; one-hot somado por atributo original",
            "valor_base": round(base, 3),
            "contribuicoes_por_atributo": {k: round(v, 3) for k, v in contrib.items()},
            "soma_reconstruida": round(float(base + valores.sum()), 3),
        },
        "importancia_global_permutacao": artefato["importancia_permutacao"],
        "avaliacao_teste": avaliacao,
        "origem": artefato["origem"],
        "versoes": {**artefato["versoes"], "shap": shap.__version__},
        "limites": LIMITES_INTERPRETACAO,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ficha", default=str(PASTA / "ficha.json"))
    args = parser.parse_args()

    artefato = carregar_artefato()
    with open(args.ficha, encoding="utf-8") as arquivo:
        ficha = validar_ficha(json.load(arquivo), artefato)
    pacote = prever(ficha, artefato)

    texto = json.dumps(pacote, ensure_ascii=False, indent=2, allow_nan=False)
    historico = PASTA_RESULTADOS / "previsoes"
    historico.mkdir(parents=True, exist_ok=True)
    nome = "r-" + datetime.now().strftime("%y%m%d-%H%M%S%f") + ".json"
    (historico / nome).write_text(texto, encoding="utf-8")
    (PASTA / "resultado.json").write_text(texto, encoding="utf-8")
    assert json.loads((PASTA / "resultado.json").read_text(encoding="utf-8")) == pacote

    p = pacote["previsao"]
    print(f"Popularidade prevista: {p['popularidade_prevista']:.1f} "
          f"(± erro médio: {p['intervalo_da_previsao'][0]}–{p['intervalo_da_previsao'][1]}, "
          f"modelo {p['modelo']})")
    print("Contribuições SHAP (pontos de popularidade):")
    print(f"  {'valor base':<18} {pacote['explicacao_shap']['valor_base']:8.2f}")
    for atributo, v in pacote["explicacao_shap"]["contribuicoes_por_atributo"].items():
        print(f"  {atributo:<18} {v:+8.2f}")
    print("RESULTADO GRAVADO: resultado.json")


if __name__ == "__main__":
    main()
