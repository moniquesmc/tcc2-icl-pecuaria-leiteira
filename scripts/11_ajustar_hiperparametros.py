"""
Ajuste de hiperparâmetros na VALIDAÇÃO (sugestão do orientador, reunião de 14/11/2026).

Nos experimentos principais, dois hiperparâmetros foram fixados:
  - KATE / Ordenação: k = 5 vizinhos
  - Few-shot: 1 exemplo por categoria (11 exemplos)
Este script testa outros valores usando SÓ a validação de cada fold (o teste não é usado
na escolha) e mostra se o valor fixado está entre os melhores.

Grade testada (pode mudar pela linha de comando):
  - KATE:     k = 1, 3, 5, 8, 10
  - Few-shot: m = 1, 2, 3 exemplos por categoria (11, 22, 33 exemplos)
    (m = 1 usa EXATAMENTE os mesmos exemplos do experimento principal; m = 2 e 3
     acrescentam exemplos a esse conjunto, então os conjuntos são encaixados)

Saída:
  dados/processados/hiperparametros/ajuste_<modelo>.json   (tudo, inclusive as respostas)
  dados/processados/hiperparametros/resumo_<modelo>.md     (tabela para o texto/slides)

Uso:
  python scripts/11_ajustar_hiperparametros.py --modelo qwen
  python scripts/11_ajustar_hiperparametros.py --modelo ollama --n-validacao 50
  python scripts/11_ajustar_hiperparametros.py --modelo qwen --avaliar-teste

  --avaliar-teste : depois de escolher o melhor valor na validação de cada fold, roda o
                    TESTE com esse valor e compara com o resultado do valor fixado.

As respostas vão para o mesmo cache dos outros scripts: se o script parar, é só rodar de
novo que ele continua de onde parou, sem repetir chamadas.
"""
import argparse
import json
import random

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

from comum import (CATEGORIAS, K_KATE, MODELOS_PARA_TESTAR, PASTA_DADOS, SEMENTE, FalhaDeAPI,
                   IndiceSemantico, arquivo_resultados, carregar_folds, carregar_resultados,
                   chamar_com_retry, extrair_classe_prevista, montar_prompt, progresso,
                   salvar_resultados, selecionar_fewshot)

PASTA_SAIDA = PASTA_DADOS / "hiperparametros"
K_PADRAO = (1, 3, 5, 8, 10)
M_PADRAO = (1, 2, 3)


def fewshot_m_por_categoria(treino, n_fold, m):
    """m exemplos por categoria. m = 1 é idêntico ao Few-shot principal; m > 1 acrescenta
    exemplos sorteados (semente fixa) ao mesmo conjunto, em ordem aleatória fixa."""
    base = selecionar_fewshot(treino, n_fold)
    if m == 1:
        return base
    rng = random.Random(SEMENTE + 500 + n_fold)
    usados = {d["id"] for d in base}
    extras = []
    for c in CATEGORIAS:
        candidatos = [t for t in treino if t["categoria_tematica"] == c and t["id"] not in usados]
        extras += rng.sample(candidatos, min(m - 1, len(candidatos)))
    demos = base + extras
    rng.shuffle(demos)
    return demos


def metricas(gabarito, previsto):
    return {"acuracia": accuracy_score(gabarito, previsto) * 100,
            "f1_macro": f1_score(gabarito, previsto, labels=CATEGORIAS, average="macro",
                                 zero_division=0) * 100}


def avaliar(perguntas, montar_demos, modelo, n_fold, rotulo):
    """Classifica uma lista de perguntas com as demonstrações dadas por montar_demos(item)."""
    previstos, brutos = [], []
    for i, item in enumerate(perguntas, 1):
        progresso(modelo, n_fold, i, len(perguntas), f"{rotulo}: ")
        resposta = chamar_com_retry(montar_prompt(item["pergunta"], montar_demos(item)), modelo)
        brutos.append(resposta)
        previstos.append(extrair_classe_prevista(resposta) or "")
    print()
    gabarito = [item["categoria_tematica"] for item in perguntas]
    return {**metricas(gabarito, previstos),
            "predicoes": [{"id_pergunta": it["id"], "gabarito": g, "previsto": p, "resposta_bruta": b}
                          for it, g, p, b in zip(perguntas, gabarito, previstos, brutos)]}


def configuracoes(ks, ms):
    return [("kate", k) for k in ks] + [("fewshot", m) for m in ms]


def nome_config(tecnica, valor):
    return f"KATE k={valor}" if tecnica == "kate" else f"Few-shot {valor}/categoria ({valor * 11} ex.)"


def processar_modelo(modelo, args):
    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    arquivo = PASTA_SAIDA / f"ajuste_{modelo}.json"
    dados = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}
    folds = [f for f in carregar_folds() if not args.folds or f["fold"] in args.folds]
    configs = configuracoes(args.k, args.m)

    for fold in folds:
        n_fold = fold["fold"]
        chave_fold = str(n_fold)
        registro = dados.setdefault(chave_fold, {"validacao": {}, "teste": {}})
        validacao = fold["validacao"][:args.n_validacao]
        indice = None

        # ---------- 1) validação: todas as configurações ----------
        for tecnica, valor in configs:
            chave = f"{tecnica}_{valor}"
            if chave in registro["validacao"]:
                continue
            if tecnica == "kate":
                indice = indice or IndiceSemantico(fold["treino"])
                montar = lambda item, k=valor: list(reversed(indice.vizinhos(item["pergunta"], k)[0]))
            else:
                demos = fewshot_m_por_categoria(fold["treino"], n_fold, valor)
                montar = lambda item, d=demos: d
            print(f"\n--- {modelo.upper()} | fold {n_fold} | validação | {nome_config(tecnica, valor)} ---")
            registro["validacao"][chave] = avaliar(validacao, montar, modelo, n_fold, "validação")
            salvar_resultados(arquivo, dados)

        # ---------- 2) teste (opcional) com o melhor valor da validação ----------
        if args.avaliar_teste:
            for tecnica, fixo in (("kate", K_KATE), ("fewshot", 1)):
                valores = [v for t, v in configs if t == tecnica]
                melhor = max(valores, key=lambda v: registro["validacao"][f"{tecnica}_{v}"]["f1_macro"])
                registro.setdefault("escolhido", {})[tecnica] = melhor
                chave = f"{tecnica}_{melhor}"
                if melhor == fixo or chave in registro["teste"]:
                    continue
                if tecnica == "kate":
                    indice = indice or IndiceSemantico(fold["treino"])
                    montar = lambda item, k=melhor: list(reversed(indice.vizinhos(item["pergunta"], k)[0]))
                else:
                    demos = fewshot_m_por_categoria(fold["treino"], n_fold, melhor)
                    montar = lambda item, d=demos: d
                print(f"\n--- {modelo.upper()} | fold {n_fold} | TESTE | {nome_config(tecnica, melhor)} ---")
                registro["teste"][chave] = avaliar(fold["teste"], montar, modelo, n_fold, "teste")
                salvar_resultados(arquivo, dados)

    escrever_resumo(modelo, dados, configs, args)


def resultado_principal_no_teste(modelo):
    """F1-macro por fold do experimento principal (k = 5 e 1 exemplo/categoria), do script 5."""
    saida = {}
    for fold in carregar_resultados(arquivo_resultados("basicos", modelo)):
        for tecnica in ("kate", "fewshot"):
            g = [p["gabarito"] for p in fold["predicoes"]]
            p = [p[f"previsto_{tecnica}"] or "" for p in fold["predicoes"]]
            saida.setdefault(tecnica, {})[str(fold["fold"])] = metricas(g, p)["f1_macro"]
    return saida


def escrever_resumo(modelo, dados, configs, args):
    folds = sorted(dados, key=int)
    linhas = [f"## Ajuste de hiperparâmetros na validação · {modelo}",
              "",
              f"F1-macro (%) na validação ({args.n_validacao} perguntas por fold), média nos folds {', '.join(folds)}.",
              "Em negrito: o valor fixado nos experimentos principais. p = teste t pareado contra o valor fixado.",
              "",
              "| Configuração | F1-macro validação | Acurácia validação | p vs. valor fixado |",
              "|---|---|---|---|"]
    for tecnica, fixo in (("kate", K_KATE), ("fewshot", 1)):
        base = [dados[f]["validacao"].get(f"{tecnica}_{fixo}", {}).get("f1_macro") for f in folds]
        for t, valor in configs:
            if t != tecnica:
                continue
            vals = [dados[f]["validacao"].get(f"{t}_{valor}") for f in folds]
            if any(v is None for v in vals):
                continue
            f1 = [v["f1_macro"] for v in vals]
            acc = [v["acuracia"] for v in vals]
            if valor == fixo or None in base or len(folds) < 2:
                p = "—"
            elif np.allclose(f1, base):
                p = "1,000"
            else:
                p = f"{stats.ttest_rel(f1, base).pvalue:.3f}".replace(".", ",")
            nome = nome_config(t, valor)
            nome = f"**{nome}**" if valor == fixo else nome
            br = lambda x: f"{x:.2f}".replace(".", ",")
            linhas.append(f"| {nome} | {br(np.mean(f1))} | {br(np.mean(acc))} | {p} |")
    if args.avaliar_teste:
        principal = resultado_principal_no_teste(modelo)
        linhas += ["", "### Teste com o valor escolhido na validação", "",
                   "| Fold | Técnica | Valor escolhido | F1 teste (escolhido) | F1 teste (fixado) |",
                   "|---|---|---|---|---|"]
        for f in folds:
            for tecnica, fixo in (("kate", K_KATE), ("fewshot", 1)):
                melhor = dados[f].get("escolhido", {}).get(tecnica)
                if melhor is None:
                    continue
                f1_fixo = principal.get(tecnica, {}).get(f)
                f1_esc = f1_fixo if melhor == fixo else dados[f]["teste"].get(f"{tecnica}_{melhor}", {}).get("f1_macro")
                fmt = lambda x: "—" if x is None else f"{x:.2f}".replace(".", ",")
                linhas.append(f"| {f} | {tecnica} | {melhor} | {fmt(f1_esc)} | {fmt(f1_fixo)} |")
    texto = "\n".join(linhas) + "\n"
    (PASTA_SAIDA / f"resumo_{modelo}.md").write_text(texto, encoding="utf-8")
    print("\n" + texto)


def main():
    parser = argparse.ArgumentParser(description="Ajuste de k (KATE) e de exemplos por categoria (Few-shot) na validação")
    parser.add_argument("--modelo", choices=MODELOS_PARA_TESTAR, action="append",
                        help="Modelo (pode repetir). Padrão: qwen.")
    parser.add_argument("--k", type=int, nargs="+", default=list(K_PADRAO), help="Valores de k do KATE")
    parser.add_argument("--m", type=int, nargs="+", default=list(M_PADRAO), help="Exemplos por categoria no Few-shot")
    parser.add_argument("--n-validacao", type=int, default=100, help="Perguntas da validação por fold (máx. 100)")
    parser.add_argument("--folds", type=int, nargs="+", help="Rodar só estes folds (ex.: --folds 1 2)")
    parser.add_argument("--avaliar-teste", action="store_true",
                        help="Rodar o teste com o melhor valor escolhido na validação")
    args = parser.parse_args()
    for modelo in args.modelo or ["qwen"]:
        try:
            processar_modelo(modelo, args)
        except FalhaDeAPI as e:
            print(f"\n[PARADO] {e}\nRode o mesmo comando de novo depois: o que já foi feito está salvo.")


if __name__ == "__main__":
    main()
