"""
Comparativo final das 9 técnicas de ICL x 4 LLMs.

Versão SEM pandas e SEM seaborn (o Windows bloqueava uma DLL do pandas). Usa apenas
numpy, scipy, scikit-learn e matplotlib. Os resultados são idênticos aos da versão anterior.

Lê dados/processados/resultados/*.json e gera:
- comparativo_9_tecnicas.csv  (média ± desvio padrão nos 5 folds)
- metricas_por_fold.csv
- comparativo_por_tipologia.csv e graficos/comparativo_por_tipologia.png
  (desempenho em perguntas Booleanas x Descritivas)
- tabela_acuracia.md e tabela_f1_macro.md: média ± IC 95%, com negrito na melhor técnica
  de cada modelo e nas que não diferem dela (teste t pareado, p > 0,05)
- significancia_acuracia.csv e significancia_f1_macro.csv: p-valores
- graficos/comparativo_f1_macro.png e graficos/comparativo_acuracia.png
- graficos/matriz_confusao_<modelo>_<tecnica>.png

Uso:
    python scripts/6_calcular_metricas.py
"""
import csv
import json
import math
import os

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score)

from comum import CATEGORIAS, PASTA_DADOS, PASTA_RESULTADOS

try:  # os gráficos são opcionais: se o matplotlib falhar, as tabelas são geradas mesmo assim
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as erro:  # noqa: BLE001
    plt = None
    print(f"[aviso] matplotlib indisponível ({erro}); os gráficos não serão gerados.")

PASTA_GRAFICOS = PASTA_DADOS / "graficos"

# chave no JSON -> nome usado nas tabelas (na ordem de apresentação)
METODOS = {
    "zeroshot": "Zero-shot",
    "fewshot": "Few-shot",
    "kate": "KATE",
    "ordenacao": "Ordenação (KATE invertido)",
    "sg_icl": "SG-ICL",
    "ape": "APE",
    "globale": "GlobalE",
    "locale": "LocalE",
    "calibracao": "Calibração",
}
# Modelos ordenados por tamanho (como sugerido pelo orientador) e nomes para as tabelas
ORDEM_MODELOS = ["ollama", "qwen", "qwen-groq", "llama-70b", "llama-nvidia", "gemini"]
NOMES_MODELOS = {
    "ollama": "Llama 3.2 3B",
    "qwen": "Qwen3.8 27B",
    "qwen-groq": "Qwen3.8 27B (Groq, parcial)",
    "llama-70b": "Llama 3.3 70B",
    "llama-nvidia": "Llama 3.2 90B",
    "gemini": "Gemini 3.1 Flash-Lite",
}
EXTRAS = {"calibracao_sem_correcao": "Few-shot c/ probabilidades (sem calibração)"}
NOMES = {**METODOS, **EXTRAS}
INVALIDAS = {"Erro de Formatação", "Erro de Conexão", ""}
CORES = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


# ----------------------------------------------------------------------------- utilidades
def ordenar_modelos(modelos):
    return sorted(modelos, key=lambda m: (ORDEM_MODELOS.index(m) if m in ORDEM_MODELOS else 99, m))


def num(valor, casas=2):
    """Número no formato brasileiro para o CSV (vírgula decimal)."""
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return ""
    if isinstance(valor, (bool, np.bool_)):
        return str(bool(valor))
    if isinstance(valor, (int, np.integer)):
        return str(int(valor))
    texto = f"{round(float(valor), casas)}"
    return texto.replace(".", ",")


def gravar_csv(caminho, cabecalho, linhas):
    with open(caminho, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(cabecalho)
        w.writerows(linhas)


def tabela_markdown(cabecalho, linhas):
    larg = [max(len(str(c)) for c in col) for col in zip(cabecalho, *linhas)]
    fmt = lambda cel: "| " + " | ".join(str(c).ljust(l) for c, l in zip(cel, larg)) + " |"
    sep = "|:" + "|:".join("-" * (l + 1) for l in larg) + "|"
    return "\n".join([fmt(cabecalho), sep] + [fmt(l) for l in linhas])


def desvio(valores):
    return float(np.std(valores, ddof=1)) if len(valores) > 1 else float("nan")


# ----------------------------------------------------------------------------- dados
def carregar_predicoes():
    """Lista de dicionários: modelo, fold, id, gabarito, metodo, previsto."""
    linhas = []
    arquivos = sorted(PASTA_RESULTADOS.glob("*.json"))
    if not arquivos:
        print(f"Nenhum resultado em {PASTA_RESULTADOS}")
    for arquivo in arquivos:
        modelo = arquivo.stem.split("_", 1)[1]
        with open(arquivo, "r", encoding="utf-8") as f:
            folds = json.load(f)
        for fold in folds:
            for p in fold["predicoes"]:
                for chave, valor in p.items():
                    if chave.startswith("previsto_"):
                        metodo = chave[len("previsto_"):]
                        if metodo in NOMES:
                            linhas.append({"modelo": modelo, "fold": fold["fold"],
                                           "id": p["id_pergunta"], "gabarito": p["gabarito"],
                                           "metodo": metodo, "previsto": valor or ""})
    return linhas


def agrupar(linhas, *chaves):
    grupos = {}
    for l in linhas:
        grupos.setdefault(tuple(l[c] for c in chaves), []).append(l)
    return grupos


def metricas(grupo):
    y = [l["gabarito"] for l in grupo]
    p = [l["previsto"] for l in grupo]
    kw = dict(labels=CATEGORIAS, average="macro", zero_division=0)
    return {
        "acuracia": accuracy_score(y, p) * 100,
        "precisao_macro": precision_score(y, p, **kw) * 100,
        "recall_macro": recall_score(y, p, **kw) * 100,
        "f1_macro": f1_score(y, p, **kw) * 100,
        "invalidas": sum(v in INVALIDAS for v in p) / len(p) * 100,
        "n": len(grupo),
    }


# ----------------------------------------------------------------------------- significância
def teste_significancia(por_fold, coluna, nome):
    """
    Para cada modelo: encontra a técnica de maior média e compara com cada uma das outras
    usando o teste t PAREADO (scipy.stats.ttest_rel) sobre os valores dos 5 folds.
    Técnicas com p > 0,05 em relação à melhor NÃO diferem significativamente dela
    e também ficam em negrito. Também calcula o intervalo de confiança de 95% (t de Student).
    """
    resultado = []
    for modelo in ordenar_modelos({m for m, _, _ in por_fold}):
        metodos = [m for m in METODOS if any(k[0] == modelo and k[1] == m for k in por_fold)]
        if not metodos:
            continue
        # só os folds presentes em todas as técnicas (equivale ao dropna do pivot)
        folds = sorted(set.intersection(*({k[2] for k in por_fold if k[0] == modelo and k[1] == m}
                                          for m in metodos)))
        if not folds:
            continue
        vet = {m: np.array([por_fold[(modelo, m, f)][coluna] for f in folds]) for m in metodos}
        medias = {m: float(v.mean()) for m, v in vet.items()}
        melhor = max(metodos, key=lambda m: medias[m])
        n = len(folds)
        for m in sorted(metodos):
            dp = float(np.std(vet[m], ddof=1)) if n > 1 else 0.0
            ic = float(stats.t.ppf(0.975, n - 1) * dp / n ** 0.5) if n > 1 else float("nan")
            if m == melhor:
                p = float("nan")
            elif n < 2 or np.abs(vet[melhor] - vet[m]).sum() == 0:
                p = 1.0
            else:
                p = float(stats.ttest_rel(vet[melhor], vet[m]).pvalue)
            resultado.append({
                "modelo": modelo, "metodo": m, "tecnica": METODOS[m], "media": medias[m],
                "dp": dp, "ic95": ic, "folds": n, "melhor_do_modelo": METODOS[melhor],
                "p_valor_vs_melhor": p, "negrito": m == melhor or (p == p and p > 0.05),
            })
    if not resultado:
        return

    campos = ["modelo", "metodo", "tecnica", "media", "dp", "ic95", "folds",
              "melhor_do_modelo", "p_valor_vs_melhor", "negrito"]
    gravar_csv(PASTA_DADOS / f"significancia_{nome}.csv", campos,
               [[r[c] if isinstance(r[c], str) else num(r[c], 4) for c in campos] for r in resultado])

    # Tabela pronta para a monografia: média ± IC95%, em negrito os empatados com o melhor
    modelos = ordenar_modelos({r["modelo"] for r in resultado})
    celula = {}
    for r in resultado:
        txt = f"{r['media']:.2f} ± {r['ic95']:.2f}"
        celula[(r["metodo"], r["modelo"])] = f"**{txt}**" if r["negrito"] else txt
    linhas = [[METODOS[m]] + [celula.get((m, mo), "") for mo in modelos]
              for m in METODOS if any(k[0] == m for k in celula)]
    texto = (f"### {nome.replace('_', ' ').title()} (%) — média ± IC 95% (5 folds)\n\n"
             "Em negrito: melhor técnica de cada modelo e as que não diferem dela "
             "(teste t pareado, p > 0,05).\n\n"
             + tabela_markdown(["tecnica"] + [NOMES_MODELOS.get(m, m) for m in modelos], linhas) + "\n")
    with open(PASTA_DADOS / f"tabela_{nome}.md", "w", encoding="utf-8") as f:
        f.write(texto)
    print("\n" + texto)

    print("p-valores (teste t pareado contra a melhor técnica de cada modelo):")
    for modelo in modelos:
        g = [r for r in resultado if r["modelo"] == modelo]
        outros = sorted((r for r in g if r["p_valor_vs_melhor"] == r["p_valor_vs_melhor"]),
                        key=lambda r: -r["media"])
        resumo = ", ".join(f"{r['tecnica']}: p={r['p_valor_vs_melhor']:.3f}" for r in outros)
        print(f"  {NOMES_MODELOS.get(modelo, modelo)} (melhor: {g[0]['melhor_do_modelo']}) -> {resumo}")


# ----------------------------------------------------------------------------- tipologia
def carregar_tipologias():
    """id da pergunta -> tipologia (Booleana / Descritiva), definida no script 3."""
    with open(PASTA_DADOS / "perguntas_estruturadas_final.json", "r", encoding="utf-8") as f:
        return {p["id"]: p.get("tipologia") or "Não classificada" for p in json.load(f)}


def metricas_subconjunto(grupo):
    """F1-macro considerando apenas as categorias presentes no gabarito do subconjunto."""
    y = [l["gabarito"] for l in grupo]
    p = [l["previsto"] for l in grupo]
    return {
        "acuracia": accuracy_score(y, p) * 100,
        "f1_macro": f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0) * 100,
        "invalidas": sum(v in INVALIDAS for v in p) / len(p) * 100,
        "n": len(grupo),
    }


def analise_por_tipologia(linhas):
    """Desempenho de cada técnica separado por tipologia (Booleana x Descritiva),
    usando as 500 perguntas de teste (os 5 folds juntos)."""
    tipos = carregar_tipologias()
    principais = [dict(l, tipologia=tipos.get(l["id"])) for l in linhas if l["metodo"] in METODOS]
    grupos = agrupar(principais, "modelo", "metodo", "tipologia")
    tab = {k: metricas_subconjunto(g) for k, g in sorted(grupos.items())}
    gravar_csv(PASTA_DADOS / "comparativo_por_tipologia.csv",
               ["modelo", "metodo", "tipologia", "acuracia", "f1_macro", "invalidas", "n", "tecnica"],
               [[mo, me, ti, num(v["acuracia"]), num(v["f1_macro"]), num(v["invalidas"]),
                 num(float(v["n"]), 1), METODOS[me]] for (mo, me, ti), v in tab.items()])

    modelos = ordenar_modelos({k[0] for k in tab})
    tipologias = sorted({k[2] for k in tab})
    colunas = [(mo, ti) for mo in modelos for ti in tipologias]
    for col, nome in [("acuracia", "ACURÁCIA"), ("f1_macro", "F1-MACRO")]:
        print(f"\n=== {nome} (%) POR TIPOLOGIA DA PERGUNTA ===")
        corpo = [[METODOS[m]] + [f"{tab[(mo, m, ti)][col]:.2f}" if (mo, m, ti) in tab else ""
                                 for mo, ti in colunas] for m in METODOS]
        print(tabela_markdown(["tecnica"] + [f"{NOMES_MODELOS.get(mo, mo)} – {ti}" for mo, ti in colunas],
                              corpo))

    if plt is None:
        return
    nlin = math.ceil(len(modelos) / 2)
    fig, eixos = plt.subplots(nlin, 2, figsize=(14.4, 4.5 * nlin), squeeze=False)
    x = np.arange(len(METODOS))
    larg = 0.8 / len(tipologias)
    for i, mo in enumerate(modelos):
        ax = eixos[i // 2][i % 2]
        for j, ti in enumerate(tipologias):
            vals = [tab.get((mo, m, ti), {}).get("f1_macro", 0) for m in METODOS]
            ax.bar(x + (j - (len(tipologias) - 1) / 2) * larg, vals, larg, label=ti, color=CORES[j])
        ax.set_title(NOMES_MODELOS.get(mo, mo))
        ax.set_ylim(0, 100)
        ax.set_ylabel("F1-macro (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(list(METODOS.values()), rotation=35, ha="right")
        ax.legend(title="Tipologia")
    for k in range(len(modelos), nlin * 2):
        eixos[k // 2][k % 2].axis("off")
    fig.suptitle("F1-macro por tipologia da pergunta (Booleana x Descritiva)", y=1.02)
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / "comparativo_por_tipologia.png", dpi=150, bbox_inches="tight")
    plt.close("all")


# ----------------------------------------------------------------------------- gráficos
def plotar_matriz_confusao(grupo, modelo, metodo):
    rotulos = CATEGORIAS + ["Inválida"]
    prev = [l["previsto"] if l["previsto"] in CATEGORIAS else "Inválida" for l in grupo]
    cm = confusion_matrix([l["gabarito"] for l in grupo], prev, labels=rotulos)[: len(CATEGORIAS)]
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    fig.colorbar(im, ax=ax)
    limite = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > limite else "black")
    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=45, ha="right")
    ax.set_yticks(range(len(CATEGORIAS)))
    ax.set_yticklabels(CATEGORIAS)
    ax.set_title(f"Matriz de Confusão - {modelo} ({METODOS.get(metodo, metodo)})")
    ax.set_ylabel("Gabarito")
    ax.set_xlabel("Previsão do modelo")
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / f"matriz_confusao_{modelo}_{metodo}.png", dpi=150)
    plt.close(fig)


def grafico_barras(por_fold, coluna, rotulo, arquivo):
    modelos = ordenar_modelos({k[0] for k in por_fold})
    metodos = [m for m in METODOS if any(k[1] == m for k in por_fold)]
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(metodos))
    larg = 0.8 / len(modelos)
    for j, mo in enumerate(modelos):
        vals = [[v[coluna] for k, v in por_fold.items() if k[0] == mo and k[1] == m] for m in metodos]
        medias = [np.mean(v) if v else 0 for v in vals]
        dps = [np.std(v, ddof=1) if len(v) > 1 else 0 for v in vals]
        ax.bar(x + (j - (len(modelos) - 1) / 2) * larg, medias, larg, yerr=dps, capsize=3,
               label=NOMES_MODELOS.get(mo, mo), color=CORES[j % len(CORES)])
    ax.set_ylabel(f"{rotulo} (%) — média ± desvio padrão (5 folds)")
    ax.set_xlabel("Técnica de ICL")
    ax.set_title(f"Comparativo das 9 técnicas de In-Context Learning — {rotulo}")
    ax.set_xticks(x)
    ax.set_xticklabels([METODOS[m] for m in metodos], rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(title="Modelo")
    fig.tight_layout()
    fig.savefig(PASTA_GRAFICOS / arquivo, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- principal
def main():
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    linhas = carregar_predicoes()
    if not linhas:
        return

    por_fold = {k: metricas(g) for k, g in agrupar(linhas, "modelo", "metodo", "fold").items()}
    ordem = list(NOMES)
    chaves = sorted(por_fold, key=lambda k: (k[0], k[1], k[2]))

    # Resumo por modelo x técnica
    resumo = {}
    for (mo, me, fo) in chaves:
        resumo.setdefault((mo, me), []).append(por_fold[(mo, me, fo)])
    linhas_resumo = []
    for (mo, me) in sorted(resumo, key=lambda k: (k[0], ordem.index(k[1]))):
        v = resumo[(mo, me)]
        col = lambda c: [x[c] for x in v]
        linhas_resumo.append({
            "modelo": mo, "metodo": me,
            "acuracia_media": np.mean(col("acuracia")), "acuracia_dp": desvio(col("acuracia")),
            "precisao_macro_media": np.mean(col("precisao_macro")),
            "recall_macro_media": np.mean(col("recall_macro")),
            "f1_macro_media": np.mean(col("f1_macro")), "f1_macro_dp": desvio(col("f1_macro")),
            "invalidas_pct": np.mean(col("invalidas")), "folds": len(v),
            "n_total": int(sum(col("n"))), "tecnica": NOMES[me],
        })
    campos = ["modelo", "metodo", "acuracia_media", "acuracia_dp", "precisao_macro_media",
              "recall_macro_media", "f1_macro_media", "f1_macro_dp", "invalidas_pct", "folds",
              "n_total", "tecnica"]
    gravar_csv(PASTA_DADOS / "comparativo_9_tecnicas.csv", campos,
               [[r[c] if isinstance(r[c], str) else num(r[c]) for c in campos] for r in linhas_resumo])
    gravar_csv(PASTA_DADOS / "metricas_por_fold.csv",
               ["modelo", "metodo", "fold", "acuracia", "precisao_macro", "recall_macro",
                "f1_macro", "invalidas", "n"],
               [[mo, me, fo] + [num(por_fold[(mo, me, fo)][c]) for c in
                                ("acuracia", "precisao_macro", "recall_macro", "f1_macro", "invalidas")]
                + [num(float(por_fold[(mo, me, fo)]["n"]), 1)] for (mo, me, fo) in chaves])

    modelos = ordenar_modelos({r["modelo"] for r in linhas_resumo})
    idx = {(r["modelo"], r["metodo"]): r for r in linhas_resumo}
    for col, nome in [("acuracia_media", "ACURÁCIA"), ("f1_macro_media", "F1-MACRO"),
                      ("invalidas_pct", "% RESPOSTAS INVÁLIDAS")]:
        print(f"\n=== {nome} (%) — média dos 5 folds ===")
        corpo = [[METODOS[m]] + [f"{idx[(mo, m)][col]:.2f}" if (mo, m) in idx else "" for mo in modelos]
                 for m in METODOS]
        print(tabela_markdown(["tecnica"] + [NOMES_MODELOS.get(m, m) for m in modelos], corpo))

    if any(k[1] == "calibracao_sem_correcao" for k in idx):
        print("\n=== Efeito da calibração (F1-macro %) ===")
        corpo = [[NOMES[m]] + [f"{idx[(mo, m)]['f1_macro_media']:.2f}" if (mo, m) in idx else ""
                               for mo in modelos] for m in ("calibracao", "calibracao_sem_correcao")]
        print(tabela_markdown(["tecnica"] + [NOMES_MODELOS.get(m, m) for m in modelos], corpo))

    incompletos = [r for r in linhas_resumo if r["metodo"] in METODOS and r["folds"] < 5]
    if incompletos:
        print("\n[aviso] Combinações com menos de 5 folds (resultado parcial):")
        for r in incompletos:
            print(f"  {r['modelo']:12s} {r['tecnica']:30s} {r['folds']} folds")

    principais = {k: v for k, v in por_fold.items() if k[1] in METODOS}
    teste_significancia(principais, "acuracia", "acuracia")
    teste_significancia(principais, "f1_macro", "f1_macro")

    analise_por_tipologia(linhas)

    if plt is not None:
        grafico_barras(principais, "f1_macro", "F1-macro", "comparativo_f1_macro.png")
        grafico_barras(principais, "acuracia", "Acurácia", "comparativo_acuracia.png")
        for (modelo, metodo), g in agrupar([l for l in linhas if l["metodo"] in METODOS],
                                           "modelo", "metodo").items():
            plotar_matriz_confusao(g, modelo, metodo)

    print(f"\nTabela completa: {PASTA_DADOS / 'comparativo_9_tecnicas.csv'}")
    print(f"Gráficos em: {PASTA_GRAFICOS}")


if __name__ == "__main__":
    main()
