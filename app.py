"""Etapa 6 · Interface de demonstração (Streamlit).

Três abas:
  1. Formulário  - escolha o gênero e os atributos, veja a previsão e o SHAP,
                   peça a explicação da LLM e converse sobre o resultado.
  2. Assistente  - descreva a música em texto; a LLM chama o modelo de ML
                   (tool calling) e explica o resultado.
  3. Sobre       - métricas dos modelos e figuras da análise.

Execução:  streamlit run app.py
(Para as funções de LLM, configure NVIDIA_API_KEY no mesmo terminal antes.)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import processar
from config import PASTA_FIGURAS, PASTA_RESULTADOS
from explorar import AZUL, TEXTO_2, VERMELHO
from prever import carregar_artefato, completar_ficha, prever, validar_ficha

TONS = ["Dó", "Dó♯/Ré♭", "Ré", "Ré♯/Mi♭", "Mi", "Fá", "Fá♯/Sol♭", "Sol",
        "Sol♯/Lá♭", "Lá", "Lá♯/Si♭", "Si"]
CONTROLES = {  # campo: (rótulo, mínimo, máximo, passo)
    "danceability": ("Dançabilidade", 0.0, 1.0, 0.01),
    "energy": ("Energia", 0.0, 1.0, 0.01),
    "valence": ("Positividade (valence)", 0.0, 1.0, 0.01),
    "acousticness": ("Acústica", 0.0, 1.0, 0.01),
    "instrumentalness": ("Instrumental", 0.0, 1.0, 0.01),
    "speechiness": ("Presença de fala", 0.0, 1.0, 0.01),
    "liveness": ("Ao vivo (liveness)", 0.0, 1.0, 0.01),
    "loudness": ("Volume médio (dB)", -40.0, 2.0, 0.1),
    "tempo": ("Andamento (BPM)", 40.0, 220.0, 1.0),
    "duracao_min": ("Duração (min)", 0.5, 12.0, 0.1),
}

st.set_page_config(page_title="Popularidade no Spotify", page_icon="🎵", layout="wide")


@st.cache_resource
def artefato_cacheado():
    return carregar_artefato()


@st.cache_resource
def cliente_llm():
    """Cliente da LLM ou None se a chave não estiver configurada."""
    try:
        return processar.configurar_cliente()
    except ValueError:
        return None


def grafico_shap(resultado):
    shap_ = resultado["explicacao_shap"]
    contrib = list(shap_["contribuicoes_por_atributo"].items())[:10][::-1]
    nomes, valores = [c[0] for c in contrib], [c[1] for c in contrib]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(nomes, valores, color=[AZUL if v >= 0 else VERMELHO for v in valores],
            height=0.65)
    for i, v in enumerate(valores):
        ax.text(v, i, f" {v:+.1f} " if v >= 0 else f" {v:+.1f} ", va="center",
                ha="left" if v >= 0 else "right", fontsize=8, color=TEXTO_2)
    ax.axvline(0, color=TEXTO_2, linewidth=0.8)
    limite = max(abs(min(valores)), abs(max(valores))) * 1.3 or 1
    ax.set_xlim(-limite, limite)
    ax.set_xlabel("pontos de popularidade (SHAP)")
    ax.set_title(f"Partindo da média {shap_['valor_base']:.1f}: o que somou e o que tirou",
                 fontsize=10, loc="left")
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def mostrar_resultado(resultado, preenchidos=None):
    p, ctx = resultado["previsao"], resultado["contexto_treino"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Popularidade prevista", f"{p['popularidade_prevista']:.1f}",
              help="Escala 0–100 do Spotify")
    c2.metric("Intervalo (± erro médio)",
              f"{p['intervalo_da_previsao'][0]:.0f} – {p['intervalo_da_previsao'][1]:.0f}",
              help=p["intervalo_da_previsao_significa"])
    c3.metric(f"Média do gênero '{resultado['atributos']['track_genre']}'",
              f"{ctx['popularidade_media_do_genero']:.1f}",
              delta=f"{p['popularidade_prevista'] - ctx['popularidade_media_do_genero']:+.1f}",
              help=ctx["posicao_do_genero"])
    figura = grafico_shap(resultado)
    st.pyplot(figura, width="content")
    plt.close(figura)
    if preenchidos:
        st.caption("Preenchidos com o valor típico do gênero: " + ", ".join(preenchidos))
    with st.expander("Resultado completo (JSON enviado à LLM)"):
        st.json(processar.resumo_para_llm(resultado))


def aviso_llm():
    st.info("Para usar a LLM, configure a variável **NVIDIA_API_KEY** no terminal "
            "e reinicie o app (veja o README).")


# ------------------------------------------------------------------ páginas
def aba_formulario(artefato):
    generos = artefato["categorias"]["track_genre"]
    tipicos = artefato["contexto_treino"]["valores_tipicos_por_genero"]
    genero = st.selectbox("Gênero", generos, index=generos.index("pop"))
    # Ao trocar de gênero, os controles voltam aos valores típicos dele.
    if st.session_state.get("genero_form") != genero:
        st.session_state.genero_form = genero
        for campo in CONTROLES:
            _, minimo, maximo, _ = CONTROLES[campo]
            st.session_state[f"f_{campo}"] = float(min(max(tipicos[genero][campo], minimo), maximo))
        for campo in ["key", "time_signature"]:
            st.session_state[f"f_{campo}"] = tipicos[genero][campo]
        for campo in ["explicit", "mode"]:
            st.session_state[f"f_{campo}"] = bool(tipicos[genero][campo])
    st.caption("Os controles começam com os valores típicos (medianas) do gênero no treino.")

    colunas = st.columns(2)
    ficha = {"track_genre": genero}
    for i, (campo, (rotulo, minimo, maximo, passo)) in enumerate(CONTROLES.items()):
        ficha[campo] = colunas[i % 2].slider(rotulo, minimo, maximo, step=passo,
                                             key=f"f_{campo}",
                                             help=artefato["descricoes"][campo])
    c1, c2, c3, c4 = st.columns(4)
    ficha["explicit"] = int(c1.toggle("Conteúdo explícito", key="f_explicit"))
    ficha["mode"] = int(c2.toggle("Modo maior", key="f_mode"))
    ficha["key"] = c3.selectbox("Tonalidade", range(12), format_func=lambda k: TONS[k],
                                key="f_key")
    ficha["time_signature"] = c4.selectbox("Compasso",
                                           artefato["categorias"]["time_signature"],
                                           format_func=lambda t: f"{t}/4",
                                           key="f_time_signature")

    if st.button("Prever popularidade", type="primary"):
        try:
            ficha = validar_ficha(ficha, artefato)
            st.session_state.resultado_form = prever(ficha, artefato)
            st.session_state.chat_form = None
        except ValueError as erro:
            st.error(str(erro))
    resultado = st.session_state.get("resultado_form")
    if not resultado:
        return
    st.divider()
    mostrar_resultado(resultado)

    st.subheader("Explicação com IA")
    llm = cliente_llm()
    if llm is None:
        aviso_llm()
        return
    cliente, modelo = llm
    if st.session_state.get("chat_form") is None and st.button("Explicar com a LLM"):
        mensagens = processar.montar_mensagens_relato(resultado)
        with st.spinner(f"Consultando {modelo}..."):
            try:
                resposta = processar.responder(cliente, modelo, mensagens).content or ""
            except Exception as erro:  # sem expor o corpo da resposta do provedor
                st.error("Chamada não concluída: " + type(erro).__name__)
                return
        mensagens.append({"role": "assistant", "content": resposta})
        st.session_state.chat_form = mensagens
    mensagens = st.session_state.get("chat_form")
    if not mensagens:
        return
    for m in mensagens[2:]:
        st.chat_message(m["role"]).markdown(m["content"])
    if pergunta := st.chat_input("Pergunte sobre esta previsão"):
        with st.spinner("Pensando..."):
            try:
                processar.responder_chat(cliente, modelo, mensagens, pergunta)
            except Exception as erro:
                st.error("Chamada não concluída: " + type(erro).__name__)
        st.rerun()


def aba_assistente(artefato):
    st.write("Descreva a música com suas palavras. A LLM converte a descrição em "
             "atributos, **executa o modelo de ML** (tool calling) e explica o resultado.")
    llm = cliente_llm()
    if llm is None:
        aviso_llm()
        return
    cliente, modelo = llm
    if "historico_assistente" not in st.session_state:
        st.session_state.historico_assistente = [
            {"role": "system", "content": processar.REGRA_ASSISTENTE}]
        st.session_state.turnos = []
    if st.button("Nova conversa"):
        del st.session_state["historico_assistente"]
        st.rerun()
    for turno in st.session_state.turnos:
        st.chat_message("user").markdown(turno["pedido"])
        with st.chat_message("assistant"):
            if turno["resultado"]:
                mostrar_resultado(turno["resultado"])
            st.markdown(turno["resposta"])
    exemplo = "Ex.: um sertanejo romântico, acústico e meio triste, com 3 minutos e meio"
    if pedido := st.chat_input(exemplo):
        with st.spinner(f"Consultando {modelo} e o modelo de ML..."):
            try:
                resposta, resultado = processar.responder_assistente(
                    cliente, modelo, st.session_state.historico_assistente, pedido,
                    artefato, processar.definir_ferramentas(artefato))
            except Exception as erro:
                st.error("Chamada não concluída: " + type(erro).__name__)
                return
        st.session_state.turnos.append({"pedido": pedido, "resposta": resposta,
                                        "resultado": resultado})
        st.rerun()


def aba_sobre(artefato):
    avaliacao = artefato["avaliacao"]
    st.write(f"**Modelo escolhido:** {artefato['nome']} · {avaliacao['divisao']} · "
             f"{avaliacao['validacao_cruzada']} · popularidade zero "
             f"{avaliacao['popularidade_zero']}.")
    caminho = PASTA_RESULTADOS / "metricas.json"
    if caminho.exists():
        metricas = json.loads(caminho.read_text(encoding="utf-8"))
        tabela = {nome: {"RMSE (validação cruzada)": m["cv_rmse"], **m["teste"]}
                  for nome, m in metricas["modelos"].items()}
        st.dataframe(tabela, width="stretch")
    for figura in sorted(PASTA_FIGURAS.glob("*.png")):
        st.image(str(figura), caption=figura.stem.replace("_", " "), width=700)


def main():
    st.title("🎵 Previsão de popularidade no Spotify")
    st.caption("Projeto final CARCDSI · Machine Learning + LLM · dados: Spotify Tracks "
               "Dataset (Kaggle)")
    try:
        artefato = artefato_cacheado()
    except (FileNotFoundError, ValueError) as erro:
        st.error(str(erro))
        st.stop()
    with st.sidebar:
        t = artefato["avaliacao"]["teste"]
        st.header("Modelo")
        st.write(f"**{artefato['nome']}**")
        st.write(f"R² = {t['R2']:.3f} · MAE = {t['MAE']:.1f} · RMSE = {t['RMSE']:.1f}")
        st.caption("Métricas no conjunto de teste (faixas nunca vistas no treino).")
        st.header("LLM")
        llm = cliente_llm()
        st.write(f"`{llm[1]}`" if llm else "não configurada")
        if os.getenv("LLM_BASE_URL"):
            st.caption(f"Serviço: {os.getenv('LLM_BASE_URL')}")
    aba1, aba2, aba3 = st.tabs(["Formulário", "Assistente (texto livre)", "Sobre o modelo"])
    with aba1:
        aba_formulario(artefato)
    with aba2:
        aba_assistente(artefato)
    with aba3:
        aba_sobre(artefato)


main()
