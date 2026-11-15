"""
Gera a apresentação em HTML com os resultados REAIS já preenchidos.

Lê dados/processados/resultados/*.json e o modelo apresentacao/template_apresentacao.html,
calcula as métricas (mesma lógica do script 6: média ± IC 95%, teste t pareado, matrizes de
confusão, tipologia, efeito da calibração) e grava:

    apresentacao/apresentacao_tcc2.html

Pode ser rodado a qualquer momento, inclusive com os experimentos em andamento: o que ainda
não tiver resultado aparece como "—" e combinações com menos de 5 folds ganham um asterisco.

Uso:
    python scripts/10_gerar_apresentacao.py
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from comum import CATEGORIAS, PASTA_DADOS, PASTA_RESULTADOS

RAIZ = Path(__file__).resolve().parent.parent
TEMPLATE = RAIZ / "apresentacao" / "template_apresentacao.html"
# A apresentação final fica na pasta TCC2/Apresentacao (fora da pasta de código)
SAIDA = RAIZ.parent / "Apresentacao" / "apresentacao_tcc2.html"

MODELOS = [("ollama", "Llama 3.2 3B"), ("qwen", "Qwen3.8 27B"),
           ("llama-70b", "Llama 3.3 70B"), ("llama-nvidia", "Llama 3.2 90B"), ("gemini", "Gemini 3.1 Flash-Lite")]
METODOS = [("zeroshot", "Zero-shot"), ("fewshot", "Few-shot"), ("kate", "KATE"),
           ("ordenacao", "Ordenação"), ("sg_icl", "SG-ICL"), ("ape", "APE"),
           ("globale", "GlobalE"), ("locale", "LocalE"), ("calibracao", "Calibração")]

# Categorias na ordem dos capítulos do livro (para a matriz de confusão)
CAPITULOS = [
    ("Cria e Recria de Bezerras e Novilhas", "Cria e Recria"),
    ("Alimentação e Manejo de Vacas e Touros", "Alimentação e Manejo"),
    ("Recursos Forrageiros", "Recursos Forrageiros"),
    ("Reprodução", "Reprodução"),
    ("Melhoramento Genético Animal", "Melhoramento Genético"),
    ("Saúde Animal", "Saúde Animal"),
    ("Mastite e Qualidade do Leite", "Mastite e Qualidade"),
    ("Produção Orgânica de Leite", "Produção Orgânica"),
    ("Gerenciamento da Atividade Leiteira", "Gerenciamento"),
    ("Bem-estar Animal", "Bem-estar Animal"),
    ("Instalações, Ambiência e Manejo de Dejetos", "Instalações e Dejetos"),
]
assert sorted(c for c, _ in CAPITULOS) == sorted(CATEGORIAS)
NOMES_CAT = [c for c, _ in CAPITULOS]


def carregar():
    """{modelo: {metodo: {fold: [(id, gabarito, previsto), ...]}}}"""
    dados = {}
    for arquivo in sorted(PASTA_RESULTADOS.glob("*.json")):
        modelo = arquivo.stem.split("_", 1)[1]
        with open(arquivo, "r", encoding="utf-8") as f:
            folds = json.load(f)
        for fold in folds:
            for p in fold["predicoes"]:
                for chave, valor in p.items():
                    if chave.startswith("previsto_"):
                        metodo = chave[len("previsto_"):]
                        (dados.setdefault(modelo, {}).setdefault(metodo, {})
                         .setdefault(fold["fold"], [])
                         .append((p["id_pergunta"], p["gabarito"], valor or "")))
    return dados


def metricas(linhas):
    y = [g for _, g, _ in linhas]
    p = [v for _, _, v in linhas]
    kw = dict(labels=CATEGORIAS, average="macro", zero_division=0)
    return (accuracy_score(y, p) * 100,
            f1_score(y, p, **kw) * 100,
            sum(v not in CATEGORIAS for v in p) / len(p) * 100,
            precision_score(y, p, **kw) * 100,
            recall_score(y, p, **kw) * 100)


def ic95(valores):
    n = len(valores)
    if n < 2:
        return None
    return float(stats.t.ppf(0.975, n - 1) * np.std(valores, ddof=1) / np.sqrt(n))


def resumo_do_modelo(por_metodo):
    """Média, IC 95% e teste t pareado (melhor x demais) para acurácia, F1-macro, precisão e recall."""
    por_fold = {m: {f: metricas(l) for f, l in folds.items()} for m, folds in por_metodo.items()}
    res = {}
    for m, folds in por_fold.items():
        vals = list(folds.values())
        res[m] = {
            "acc": float(np.mean([v[0] for v in vals])), "acc_ic": ic95([v[0] for v in vals]),
            "f1": float(np.mean([v[1] for v in vals])), "f1_ic": ic95([v[1] for v in vals]),
            "inv": float(np.mean([v[2] for v in vals])), "folds": len(vals),
            "prec": float(np.mean([v[3] for v in vals])), "prec_ic": ic95([v[3] for v in vals]),
            "rec": float(np.mean([v[4] for v in vals])), "rec_ic": ic95([v[4] for v in vals]),
        }
    for idx, nome in ((0, "acc"), (1, "f1"), (3, "prec"), (4, "rec")):
        metodos = [m for m, _ in METODOS if m in por_fold]
        comuns = sorted(set.intersection(*(set(por_fold[m]) for m in metodos))) if metodos else []
        if not comuns:
            continue
        vetor = {m: np.array([por_fold[m][f][idx] for f in comuns]) for m in metodos}
        melhor = max(metodos, key=lambda m: vetor[m].mean())
        for m in metodos:
            if m == melhor:
                p = None
            elif len(comuns) < 2 or np.allclose(vetor[melhor], vetor[m]):
                p = 1.0
            else:
                p = float(stats.ttest_rel(vetor[melhor], vetor[m]).pvalue)
                p = 0.0 if np.isnan(p) else p
            res[m][f"p_{nome}"] = p
            res[m][f"neg_{nome}"] = bool(m == melhor or (p is not None and p > 0.05))
    return res


def matriz(linhas):
    """11 linhas (verdadeira) x 12 colunas (11 previstas + inválida), na ordem dos capítulos."""
    idx = {c: i for i, c in enumerate(NOMES_CAT)}
    M = [[0] * (len(NOMES_CAT) + 1) for _ in NOMES_CAT]
    for _, g, p in linhas:
        M[idx[g]][idx.get(p, len(NOMES_CAT))] += 1
    return M


def tipologia(linhas, tipos):
    out = {}
    for tipo in ("Booleana", "Descritiva"):
        sub = [x for x in linhas if tipos.get(x[0]) == tipo]
        if sub:
            y = [g for _, g, _ in sub]
            p = [v for _, _, v in sub]
            rot = sorted(set(y))
            out[tipo] = {"acc": accuracy_score(y, p) * 100,
                         "f1": f1_score(y, p, labels=rot, average="macro", zero_division=0) * 100,
                         "n": len(sub)}
    return out


def main():
    dados = carregar()
    with open(PASTA_DADOS / "perguntas_estruturadas_final.json", "r", encoding="utf-8") as f:
        tipos = {p["id"]: p.get("tipologia") for p in json.load(f)}

    saida = {
        "gerado_em": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "categorias": [{"num": i + 1, "nome": c, "curto": s} for i, (c, s) in enumerate(CAPITULOS)],
        "modelos": [{"id": m, "nome": n} for m, n in MODELOS if m in dados],
        "metodos": [{"id": m, "nome": n} for m, n in METODOS],
        "resumo": {}, "matrizes": {}, "tipologia": {}, "calibracao": {},
    }
    for modelo, por_metodo in dados.items():
        principais = {m: f for m, f in por_metodo.items() if m in dict(METODOS)}
        saida["resumo"][modelo] = resumo_do_modelo(principais)
        todas = {m: [x for fl in f.values() for x in fl] for m, f in principais.items()}
        saida["matrizes"][modelo] = {m: matriz(l) for m, l in todas.items()}
        saida["tipologia"][modelo] = {m: tipologia(l, tipos) for m, l in todas.items()}
        if "calibracao" in por_metodo and "calibracao_sem_correcao" in por_metodo:
            f1 = lambda folds: float(np.mean([metricas(l)[1] for l in folds.values()]))
            saida["calibracao"][modelo] = {"sem": f1(por_metodo["calibracao_sem_correcao"]),
                                           "com": f1(por_metodo["calibracao"])}

    html = TEMPLATE.read_text(encoding="utf-8")
    marcador = "/*__DADOS__*/null"
    if marcador not in html:
        raise SystemExit("Marcador de dados não encontrado no template.")
    dados_js = json.dumps(saida, ensure_ascii=False).replace("</", "<\\/")
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(html.replace(marcador, dados_js), encoding="utf-8")

    print(f"Apresentação gerada: {SAIDA}")
    for m in saida["modelos"]:
        r = saida["resumo"].get(m["id"], {})
        folds = {k: v["folds"] for k, v in r.items()}
        print(f"  {m['nome']}: técnicas com resultado = {len(r)}/9 · folds por técnica = {folds}")
    print("Abra o arquivo no navegador (Chrome/Edge). Setas ← → para navegar, F para tela cheia.")


if __name__ == "__main__":
    main()
