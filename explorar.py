"""Etapa 2 · Análise exploratória (EDA).

Lê dados/processado/spotify_limpo.csv (gerado por preparar.py) e grava as
figuras em resultados/figuras/ e um resumo em resultados/eda.json.

Execução:  python explorar.py
"""
import json

import matplotlib
matplotlib.use("Agg")  # gera arquivos sem abrir janelas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from config import ALVO, DADOS_LIMPOS, NUMERICAS, PASTA_FIGURAS, PASTA_RESULTADOS

# Paleta (azul = série principal; laranja = destaque; vermelho/azul = divergente)
AZUL, LARANJA, VERMELHO = "#2a78d6", "#eb6834", "#e34948"
CINZA_NEUTRO, TEXTO, TEXTO_2, GRADE = "#f0efec", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGENTE = LinearSegmentedColormap.from_list(
    "azul_vermelho", [VERMELHO, CINZA_NEUTRO, AZUL])

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": TEXTO_2,
    "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": "white",
})


def salvar(fig, nome):
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    caminho = PASTA_FIGURAS / nome
    fig.savefig(caminho)
    plt.close(fig)
    print("FIGURA:", caminho.relative_to(PASTA_RESULTADOS.parent))


def histograma_popularidade(dados):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    zeros = (dados[ALVO] == 0).sum()
    ax.hist(dados.loc[dados[ALVO] > 0, ALVO], bins=range(1, 102), color=AZUL,
            edgecolor="white", linewidth=0.3, label="popularidade > 0")
    ax.bar(0.5, zeros, width=1, color=LARANJA, label="popularidade = 0")
    ax.annotate(f"{zeros:,} faixas com 0\n({zeros / len(dados):.1%})".replace(",", "."),
                xy=(1.2, zeros), xytext=(10, zeros * 0.92), fontsize=8,
                color=TEXTO, arrowprops={"arrowstyle": "-", "color": TEXTO_2})
    ax.set(title="Distribuição da popularidade", xlabel="popularity (0 a 100)",
           ylabel="número de faixas", xlim=(-1, 101))
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper right")
    salvar(fig, "01_distribuicao_popularidade.png")


def popularidade_por_genero(dados, n=15):
    mediana = dados.groupby("track_genre")[ALVO].median().sort_values()
    extremos = pd.concat([mediana.head(n), mediana.tail(n)])
    fig, ax = plt.subplots(figsize=(7, 7))
    cores = [LARANJA] * n + [AZUL] * n
    ax.barh(extremos.index, extremos.values, color=cores, height=0.7)
    for i, v in enumerate(extremos.values):
        ax.text(v + 0.8, i, f"{v:.0f}", va="center", fontsize=7, color=TEXTO_2)
    ax.axhline(n - 0.5, color=TEXTO_2, linewidth=0.6, linestyle="--")
    ax.set(title=f"Mediana da popularidade: {n} gêneros mais e {n} menos populares",
           xlabel="mediana de popularity (sem zeros)")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    salvar(fig, "02_popularidade_por_genero.png")
    return mediana


def zeros_por_genero(dados, n=20):
    fracao = (dados.assign(z=dados[ALVO] == 0).groupby("track_genre")["z"]
              .mean().sort_values().tail(n) * 100)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(fracao.index, fracao.values, color=LARANJA, height=0.7)
    for i, v in enumerate(fracao.values):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=7, color=TEXTO_2)
    ax.set(title=f"Gêneros com maior proporção de popularidade zero (top {n})",
           xlabel="% das faixas do gênero com popularity = 0")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    salvar(fig, "03_zeros_por_genero.png")
    return fracao


def correlacao_com_alvo(dados):
    corr = (dados[NUMERICAS + ["explicit", "mode", ALVO]]
            .corr(method="spearman")[ALVO].drop(ALVO).sort_values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(corr.index, corr.values, height=0.7,
            color=[AZUL if v > 0 else VERMELHO for v in corr.values])
    for i, v in enumerate(corr.values):
        ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7, color=TEXTO_2)
    limite = max(abs(corr.values)) * 1.35
    ax.set(xlim=(-limite, limite), xlabel="correlação de Spearman com popularity",
           title="Correlação de cada atributo com a popularidade")
    ax.axvline(0, color=TEXTO_2, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    salvar(fig, "04_correlacao_com_popularidade.png")
    return corr


def mapa_correlacao(dados):
    colunas = NUMERICAS + [ALVO]
    corr = dados[colunas].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 6))
    imagem = ax.imshow(corr, cmap=DIVERGENTE, vmin=-1, vmax=1)
    ax.set_xticks(range(len(colunas)), colunas, rotation=45, ha="right")
    ax.set_yticks(range(len(colunas)), colunas)
    ax.grid(False)
    for i in range(len(colunas)):
        for j in range(len(colunas)):
            v = corr.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 else TEXTO)
    fig.colorbar(imagem, ax=ax, shrink=0.8, label="Spearman")
    ax.set_title("Correlação entre os atributos numéricos")
    salvar(fig, "05_mapa_correlacao.png")
    return corr


def main():
    dados = pd.read_csv(DADOS_LIMPOS)
    sem_zeros = dados[dados[ALVO] > 0]
    print(f"Faixas: {len(dados)} | sem zeros: {len(sem_zeros)}")

    histograma_popularidade(dados)
    mediana = popularidade_por_genero(sem_zeros)
    fracao_zeros = zeros_por_genero(dados)
    corr_alvo = correlacao_com_alvo(sem_zeros)
    corr = mapa_correlacao(sem_zeros)

    # Quanto da variância o gênero explica sozinho (R² de prever a média do gênero)
    media_genero = sem_zeros.groupby("track_genre")[ALVO].transform("mean")
    r2_genero = 1 - ((sem_zeros[ALVO] - media_genero) ** 2).sum() / \
        ((sem_zeros[ALVO] - sem_zeros[ALVO].mean()) ** 2).sum()

    # Só o triângulo superior (sem diagonal), para não repetir cada par.
    superior = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pares = corr.where(superior).stack().abs().sort_values(ascending=False)
    resumo = {
        "faixas": len(dados), "faixas_sem_zeros": len(sem_zeros),
        "r2_media_por_genero_sem_zeros": round(float(r2_genero), 4),
        "correlacao_spearman_com_popularidade": corr_alvo.round(4).to_dict(),
        "pares_mais_correlacionados": {f"{a} x {b}": round(float(v), 3)
            for (a, b), v in pares.head(5).items()},
        "generos_maior_mediana": mediana.tail(5).round(1).to_dict(),
        "generos_menor_mediana": mediana.head(5).round(1).to_dict(),
        "generos_mais_zeros_percentual": fracao_zeros.tail(5).round(1).to_dict(),
        "popularidade_media_por_explicit": sem_zeros.groupby("explicit")[ALVO]
            .mean().round(2).to_dict(),
        "popularidade_media_por_mode": sem_zeros.groupby("mode")[ALVO]
            .mean().round(2).to_dict(),
        # Faixas com zero têm áudio parecido com as demais? (evidência para D3)
        "media_atributos_zero_vs_demais": pd.DataFrame({
            "popularidade_zero": dados.loc[dados[ALVO] == 0, NUMERICAS].mean(),
            "popularidade_maior_que_zero": sem_zeros[NUMERICAS].mean(),
        }).round(3).to_dict(orient="index"),
    }
    (PASTA_RESULTADOS / "eda.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
