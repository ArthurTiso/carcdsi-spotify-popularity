"""Avalia a fidelidade numérica das explicações da LLM.

Gera N relatos por configuração, alternando entre cinco músicas (ficha.json e
fichas típicas de samba, metal, classical e k-pop), e confere em cada um se todos os
números citados pela LLM existem no JSON enviado a ela (aceitando
arredondamento, números escritos em textos do JSON e as diferenças
previsão − média do gênero / média geral / valor base). Também conta quantas
respostas trazem marcação LaTeX.

Compara três configurações:
  prompt_original  primeira versão do prompt ("Use os números dos JSONs;
                   arredonde só na apresentação."), JSON com todas as casas,
                   temperatura padrão do serviço, sem limpeza de LaTeX
  prompt_revisado  prompt atual ("cite os números exatamente como aparecem"),
                   mesmo JSON completo, temperatura padrão, sem limpeza
  final            o que processar.py usa: prompt atual + números arredondados
                   no JSON + temperatura 0 + limpeza de LaTeX

Execução (precisa de NVIDIA_API_KEY):
  python avaliar_llm.py              # 10 relatos por configuração
  python avaliar_llm.py --n 20
Resultado: resultados/avaliacao_llm.json
"""
import argparse
import json
import math
import re
import time

import processar
from config import DESCRICOES, PASTA, PASTA_RESULTADOS
from prever import carregar_artefato, completar_ficha, prever, validar_ficha

NUMERO = re.compile(r"\d+(?:[.,]\d+)?")
TRIVIAIS = {0, 1, 2, 3, 4, 5, 10, 100}
CONFIGURACOES = ["prompt_original", "prompt_revisado", "final"]
GENEROS_EXTRAS = ["samba", "metal", "classical", "k-pop"]


def numeros_do_json(objeto, saida):
    if isinstance(objeto, bool):
        return
    if isinstance(objeto, (int, float)):
        saida.append(abs(float(objeto)))
    elif isinstance(objeto, str):
        saida.extend(float(x.replace(",", ".")) for x in NUMERO.findall(objeto))
    elif isinstance(objeto, dict):
        for v in objeto.values():
            numeros_do_json(v, saida)
    elif isinstance(objeto, list):
        for v in objeto:
            numeros_do_json(v, saida)


def numeros_validos(resumo):
    validos = []
    numeros_do_json(resumo, validos)
    p = resumo["previsao"]["popularidade_prevista"]
    ctx = resumo["contexto_treino"]
    validos += [abs(p - ctx["popularidade_media_do_genero"]),
                abs(p - ctx["popularidade_media_geral"]),
                abs(p - resumo["explicacao_shap"]["valor_base"])]
    return validos


def numero_confere(x, validos):
    if x in TRIVIAIS:
        return True
    casas = len(repr(x).split(".")[1]) if "." in repr(x) and not repr(x).endswith(".0") else 0
    fator = 10 ** casas
    return any(abs(x - v) < 1e-9 or math.floor(v * fator) / fator == x
               or math.ceil(v * fator) / fator == x for v in validos)


FRASE_ORIGINAL = "Use os números dos JSONs; arredonde só na apresentação."
FRASE_REVISADA = ("Cite os números exatamente como aparecem no JSON (já estão arredondados);\n"
                  "não recalcule a previsão nem mude casas decimais.")


def mensagens_sem_melhorias(resultado, prompt_original):
    """JSON completo (sem arredondar), com o prompt original ou o revisado."""
    regra = processar.REGRA_RELATO
    if prompt_original:
        if FRASE_REVISADA not in regra:
            raise RuntimeError("O prompt de processar.py mudou; atualize FRASE_REVISADA.")
        regra = regra.replace(FRASE_REVISADA, FRASE_ORIGINAL)
    resumo = {k: resultado[k] for k in ("atributos", "previsao", "contexto_treino",
                                        "explicacao_shap", "limites")} | {
        "avaliacao_teste": {k: resultado["avaliacao_teste"][k]
                            for k in ("modelo", "teste", "referencia_media_genero")},
        "descricao_atributos": DESCRICOES}
    conteudo = ("RESULTADO DO MODELO DE ML E EXPLICAÇÃO SHAP (JSON):\n"
                + json.dumps(resumo, ensure_ascii=False, indent=2)
                + "\n\nPERGUNTA DA PESSOA:\n" + processar.PERGUNTA_RELATO)
    return [{"role": "system", "content": regra},
            {"role": "user", "content": conteudo}]


def avaliar(cliente, modelo, resultados, n, configuracao):
    """n relatos, alternando entre as músicas de `resultados`."""
    melhorias = configuracao == "final"
    linhas = []
    for i in range(n):
        resultado = resultados[i % len(resultados)]
        validos = numeros_validos(processar.resumo_para_llm(resultado) if melhorias
                                  else {k: resultado[k] for k in resultado})
        inicio = time.time()
        if melhorias:
            texto = processar.responder(
                cliente, modelo, processar.montar_mensagens_relato(resultado)).content
        else:
            retorno = cliente.chat.completions.create(
                model=modelo, messages=mensagens_sem_melhorias(
                    resultado, prompt_original=configuracao == "prompt_original"))
            texto = retorno.choices[0].message.content
        texto = texto or ""
        citados = [float(x.replace(",", ".")) for x in NUMERO.findall(texto)]
        errados = sorted({x for x in citados if not numero_confere(x, validos)})
        latex = "$" in texto or "\\" in texto
        genero = resultado["atributos"]["track_genre"]
        linhas.append({"genero": genero, "numeros_citados": len(citados),
                       "numeros_fora_do_json": errados,
                       "latex": latex, "segundos": round(time.time() - inicio, 1)})
        print(f"  #{i + 1:<2} {genero:<10} {len(citados):>3} números citados | fora do JSON: "
              f"{errados or '-'} | LaTeX: {'sim' if latex else 'não'}")
    return {
        "respostas": n,
        "respostas_com_numero_fora_do_json": sum(bool(l["numeros_fora_do_json"]) for l in linhas),
        "respostas_com_latex": sum(l["latex"] for l in linhas),
        "tempo_medio_s": round(sum(l["segundos"] for l in linhas) / n, 1),
        "detalhes": linhas,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10, help="relatos por configuração")
    args = parser.parse_args()

    # ficha.json (pop) + fichas típicas de outros gêneros, para variar os casos
    _, resultado = processar.ler_contexto()
    artefato = carregar_artefato()
    resultados = [resultado]
    for genero in GENEROS_EXTRAS:
        ficha, _ = completar_ficha({"track_genre": genero}, artefato)
        resultados.append(prever(validar_ficha(ficha, artefato), artefato))
    cliente, modelo = processar.configurar_cliente()
    saida = {"modelo_llm": modelo,
             "musicas": ["ficha.json (pop)"] + [f"típica de {g}" for g in GENEROS_EXTRAS]}
    try:
        for nome in CONFIGURACOES:
            print(f"\n{nome.replace('_', ' ').upper()} ({modelo}):")
            saida[nome] = avaliar(cliente, modelo, resultados, args.n, nome)
    finally:
        cliente.close()

    print("\nResumo:")
    for nome in CONFIGURACOES:
        r = saida[nome]
        print(f"  {nome:<16} número fora do JSON: {r['respostas_com_numero_fora_do_json']}/"
              f"{r['respostas']} | LaTeX: {r['respostas_com_latex']}/{r['respostas']} | "
              f"{r['tempo_medio_s']} s por resposta")
    caminho = PASTA_RESULTADOS / "avaliacao_llm.json"
    caminho.write_text(json.dumps(saida, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GRAVADO:", caminho.relative_to(PASTA))


if __name__ == "__main__":
    main()
