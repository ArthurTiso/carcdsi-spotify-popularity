"""Etapa 1 · Preparar os dados antes de treinar.

Lê o CSV original do Kaggle (dados/bruto/dataset.csv), aplica a limpeza e
grava dados/processado/spotify_limpo.csv, além de um relatório com a
quantidade de linhas em cada etapa (resultados/preparacao.json).

Decisões de limpeza (justificadas em docs/decisoes.md):
  1. Remove a coluna "Unnamed: 0" (índice sem valor preditivo).
  2. Remove linhas sem artista/álbum/nome (1 linha).
  3. Remove linhas 100% duplicadas.
  4. Consolida o mesmo track_id listado em vários gêneros em UMA linha:
     os atributos de áudio são idênticos; o gênero é sorteado entre os
     rótulos da faixa (semente 42) e a popularidade é a maior registrada.
  5. tempo = 0 e time_signature 0/1 são estimativas inválidas -> ausentes
     (preenchidos depois, dentro do pipeline de treino).
  6. Cria duracao_min, converte explicit para 0/1 e cria musica_id
     (artista + nome normalizados) e grupo_id, que une as versões da mesma
     música (mesmo musica_id) E as gravações com áudio idêntico publicadas
     com outro título ("feat.", "From ...", tradução). grupo_id é usado para
     manter a mesma música inteira no treino ou no teste. Cria também
     artista_principal (primeiro artista), usado em um experimento.
  7. Marca popularidade zero (popularidade_zero = 1). Essas faixas NÃO são
     apagadas aqui: treinar.py decide se as usa (ver docs/decisoes.md).

Execução:  python preparar.py
"""
import json

import pandas as pd

from config import (ALVO, ARQUIVO_ORIGEM, DADOS_BRUTOS, DADOS_LIMPOS,
                    RELATORIO_PREPARACAO, SEMENTE)


def normalizar_texto(serie):
    """Minúsculas e espaços simples, para comparar nomes de artistas/músicas."""
    return serie.str.lower().str.strip().str.replace(r"\s+", " ", regex=True)


ATRIBUTOS_AUDIO = ["danceability", "energy", "loudness", "speechiness",
                   "acousticness", "instrumentalness", "liveness", "valence",
                   "tempo", "duracao_min", "key", "mode"]


def agrupar_mesma_musica(dados):
    """Une musica_id e gravações de áudio idêntico (union-find).

    Duas faixas ficam no mesmo grupo se têm o mesmo artista + nome OU
    exatamente os mesmos atributos de áudio; a relação é transitiva.
    """
    assinatura = (dados[ATRIBUTOS_AUDIO].fillna(-1).round(6).astype(str)
                  .agg("|".join, axis=1))
    pai = {}

    def raiz(no):
        while pai.setdefault(no, no) != no:
            pai[no] = pai[pai[no]]
            no = pai[no]
        return no

    for musica, audio in zip(dados["musica_id"], assinatura):
        a, b = raiz("m:" + musica), raiz("a:" + audio)
        if a != b:
            pai[a] = b
    raizes = pd.Series([raiz("m:" + m) for m in dados["musica_id"]], index=dados.index)
    return pd.factorize(raizes)[0]


def preparar(caminho=DADOS_BRUTOS, semente=SEMENTE, mostrar=True):
    """Limpa o CSV original. `semente` controla o sorteio do gênero (passo 3)."""
    etapas = []

    def registrar(nome, tabela):
        etapas.append({"etapa": nome, "linhas": len(tabela)})
        if mostrar:
            print(f"{nome:<55} {len(tabela):>7} linhas")

    dados = pd.read_csv(caminho)
    registrar("0. CSV original", dados)

    dados = dados.drop(columns=["Unnamed: 0"])
    dados = dados.dropna(subset=["artists", "album_name", "track_name"])
    registrar("1. Sem valores ausentes em artista/álbum/nome", dados)

    dados = dados.drop_duplicates()
    registrar("2. Sem linhas 100% duplicadas", dados)

    # 3. Mesmo track_id em vários gêneros: só popularity e track_genre variam.
    colunas_fixas = [c for c in dados.columns
                     if c not in ("track_id", "track_genre", ALVO)]
    variacao = dados.groupby("track_id")[colunas_fixas].nunique().max()
    if (variacao > 1).any():
        raise ValueError(f"Atributos variam no mesmo track_id: {variacao[variacao > 1]}")
    embaralhados = dados.sample(frac=1, random_state=semente)
    n_generos = embaralhados.groupby("track_id")["track_genre"].nunique()
    dados = (embaralhados
             .groupby("track_id", as_index=False, sort=False)
             .agg({**{c: "first" for c in colunas_fixas},
                   "track_genre": "first", ALVO: "max"}))
    registrar("3. Um registro por track_id (gêneros consolidados)", dados)

    # 4. Estimativas inválidas viram ausentes (imputadas no pipeline de treino).
    tempo_zero = int((dados["tempo"] == 0).sum())
    compasso_invalido = int(dados["time_signature"].isin([0, 1]).sum())
    dados["tempo"] = dados["tempo"].mask(dados["tempo"] == 0)
    dados["time_signature"] = (dados["time_signature"]
                               .mask(dados["time_signature"].isin([0, 1]))
                               .astype("Int64"))

    # 5. Novas colunas e tipos.
    dados["duracao_min"] = dados["duration_ms"] / 60_000
    dados["explicit"] = dados["explicit"].astype(int)
    dados["musica_id"] = (normalizar_texto(dados["artists"]) + " || "
                          + normalizar_texto(dados["track_name"]))
    dados["popularidade_zero"] = (dados[ALVO] == 0).astype(int)
    dados["grupo_id"] = agrupar_mesma_musica(dados)
    dados["artista_principal"] = normalizar_texto(dados["artists"].str.split(";").str[0])

    ordem = ["track_id", "musica_id", "grupo_id", "artista_principal", "artists", "album_name", "track_name",
             ALVO, "popularidade_zero", "track_genre", "duracao_min",
             "explicit", "danceability", "energy", "key", "loudness", "mode",
             "speechiness", "acousticness", "instrumentalness", "liveness",
             "valence", "tempo", "time_signature"]
    dados = dados[ordem]

    zeros = dados[dados["popularidade_zero"] == 1]
    registrar("4. Sem popularidade zero (conjunto usado no modelo principal)",
              dados[dados["popularidade_zero"] == 0])

    relatorio = {
        "etapas": etapas,
        "track_ids_em_mais_de_um_genero": int((n_generos > 1).sum()),
        "musicas_unicas_artista_nome": int(dados["musica_id"].nunique()),
        "musicas_com_mais_de_um_track_id": int(
            (dados.groupby("musica_id")["track_id"].nunique() > 1).sum()),
        "grupos_mesma_musica_ou_mesmo_audio": int(dados["grupo_id"].nunique()),
        "grupos_unindo_titulos_diferentes_com_audio_identico": int(
            (dados.groupby("grupo_id")["musica_id"].nunique() > 1).sum()),
        "artistas_principais": int(dados["artista_principal"].nunique()),
        "tempo_zero_para_ausente": tempo_zero,
        "time_signature_0_ou_1_para_ausente": compasso_invalido,
        "popularidade_zero": {
            "faixas": len(zeros),
            "percentual": round(100 * len(zeros) / len(dados), 2),
            "mesma_musica_tem_outra_versao_com_popularidade": int(
                zeros["musica_id"].isin(
                    dados.loc[dados[ALVO] > 0, "musica_id"]).sum()),
            "artista_tem_outras_faixas_com_popularidade": int(
                zeros["artists"].isin(
                    dados.loc[dados[ALVO] > 0, "artists"]).sum()),
        },
        "popularidade_sem_zeros": dados.loc[dados[ALVO] > 0, ALVO]
            .describe().round(2).to_dict(),
        "generos": int(dados["track_genre"].nunique()),
    }
    return dados, relatorio


def main():
    dados, relatorio = preparar()
    DADOS_LIMPOS.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(DADOS_LIMPOS, index=False)
    RELATORIO_PREPARACAO.parent.mkdir(parents=True, exist_ok=True)
    RELATORIO_PREPARACAO.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")

    origem = {
        "nome": "Spotify Tracks Dataset",
        "fonte": "https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset",
        "autor": "Maharshi Pandya (Kaggle)",
        "linhas_originais": relatorio["etapas"][0]["linhas"],
        "linhas_apos_limpeza": len(dados),
    }
    ARQUIVO_ORIGEM.write_text(
        json.dumps(origem, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nARQUIVOS GRAVADOS:", DADOS_LIMPOS, RELATORIO_PREPARACAO,
          ARQUIVO_ORIGEM, sep="\n  ")
    print("\nPrimeiras linhas:")
    print(dados.head().to_string(index=False))
    print("\nValores ausentes (serão preenchidos no pipeline de treino):")
    ausentes = dados.isna().sum()
    print(ausentes[ausentes > 0].to_string())
    print("\nResumo:")
    print(json.dumps({k: v for k, v in relatorio.items() if k != "etapas"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
