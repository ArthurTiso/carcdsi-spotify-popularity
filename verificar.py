"""Verificação rápida da instalação e do pipeline.

Confere, em ordem: versões das bibliotecas, dados, modelo treinado, previsão
com SHAP, validação da ficha e (opcional) a conexão com a LLM.
Não altera nenhum arquivo.

Execução:
  python verificar.py          # sem chamar a LLM
  python verificar.py --llm    # também testa a chave NVIDIA_API_KEY
"""
import argparse
import importlib.metadata
import json
import sys

from config import ARQUIVO_MODELO, DADOS_BRUTOS, DADOS_LIMPOS, PASTA

falhas = []


def checar(descricao, funcao):
    try:
        detalhe = funcao()
        print(f"[ OK ] {descricao}" + (f" — {detalhe}" if detalhe else ""))
        return True
    except Exception as erro:  # noqa: BLE001 - queremos relatar qualquer falha
        print(f"[FALHA] {descricao} — {type(erro).__name__}: {erro}")
        falhas.append(descricao)
        return False


def versoes():
    if sys.version_info[:2] != (3, 14):
        print(f"        aviso: Python {sys.version.split()[0]}; o projeto usa 3.14")
    diferentes = []
    for linha in (PASTA / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "==" not in linha:
            continue
        pacote, esperada = linha.split("==")
        instalada = importlib.metadata.version(pacote)
        if instalada != esperada:
            diferentes.append(f"{pacote} {instalada} (esperado {esperada})")
    if diferentes:
        raise RuntimeError("; ".join(diferentes))
    return "todas iguais ao requirements.txt"


def dados():
    if not DADOS_BRUTOS.exists():
        raise FileNotFoundError(DADOS_BRUTOS)
    if not DADOS_LIMPOS.exists():
        raise FileNotFoundError(f"{DADOS_LIMPOS} — execute preparar.py")
    return "bruto e processado presentes"


def modelo_existe():
    if not ARQUIVO_MODELO.exists():
        raise FileNotFoundError(f"{ARQUIVO_MODELO} — execute treinar.py")
    return str(ARQUIVO_MODELO.relative_to(PASTA))


def previsao():
    from prever import carregar_artefato, prever, validar_ficha
    artefato = carregar_artefato()
    ficha = validar_ficha(json.loads((PASTA / "ficha.json").read_text(encoding="utf-8")),
                          artefato)
    resultado = prever(ficha, artefato)
    shap_ = resultado["explicacao_shap"]
    p = resultado["previsao"]["popularidade_prevista"]
    if not 0 <= p <= 100:
        raise ValueError(f"previsão fora de 0..100: {p}")
    return (f"{artefato['nome']}: popularidade {p:.1f}; SHAP base "
            f"{shap_['valor_base']:.1f} + contribuições = {shap_['soma_reconstruida']:.1f}")


def validacao():
    from prever import carregar_artefato, validar_ficha
    artefato = carregar_artefato()
    ficha = json.loads((PASTA / "ficha.json").read_text(encoding="utf-8"))
    ruins = [{**ficha, "energy": 2}, {**ficha, "track_genre": "inexistente"},
             {**ficha, "explicit": True}, {k: v for k, v in ficha.items() if k != "tempo"}]
    for ruim in ruins:
        try:
            validar_ficha(ruim, artefato)
        except ValueError:
            continue
        raise AssertionError(f"ficha inválida aceita: {ruim}")
    return f"{len(ruins)} fichas inválidas recusadas"


def llm():
    import processar
    cliente, modelo = processar.configurar_cliente()
    try:
        resposta = processar.responder(
            cliente, modelo, [{"role": "user", "content": "Responda apenas: ok"}],
            max_completion_tokens=300).content or ""
    finally:
        cliente.close()
    return f"{modelo} respondeu: {resposta.strip()[:40]!r}"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", action="store_true", help="testa também a chamada à LLM")
    args = parser.parse_args()

    checar("Bibliotecas", versoes)
    checar("Dados", dados)
    if checar("Modelo treinado", modelo_existe):
        checar("Previsão + SHAP (ficha.json)", previsao)
        checar("Validação da ficha", validacao)
    if args.llm:
        checar("LLM (NVIDIA)", llm)
    else:
        print("[ -- ] LLM não testada (use --llm)")

    print("\nTudo certo." if not falhas else f"\n{len(falhas)} verificação(ões) falharam.")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
