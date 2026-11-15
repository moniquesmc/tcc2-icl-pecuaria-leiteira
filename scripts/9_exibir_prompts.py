"""
Mostra o PROMPT FINAL que cada uma das 9 técnicas enviou ao LLM, junto com a resposta bruta
do modelo e a categoria extraída. Serve para entender como cada método funciona e para a
análise de erros (por que o método errou naquela pergunta).

Os prompts são reconstruídos exatamente como nos scripts 5, 7 e 8 (mesmas sementes e
funções de comum.py). Para SG-ICL, APE, GlobalE e LocalE, as partes geradas pelo LLM
(exemplos gerados, instrução escolhida, ordem escolhida) são lidas dos arquivos de resultado.

Uso:
  # todas as técnicas para uma pergunta
  python scripts/9_exibir_prompts.py --modelo ollama --id 290

  # 1 pergunta de cada categoria (verificação sugerida pelo orientador)
  python scripts/9_exibir_prompts.py --modelo ollama --uma-por-categoria

  # casos de ERRO de uma técnica (análise de erros)
  python scripts/9_exibir_prompts.py --modelo ollama --erros kate --n 5

Saída: dados/processados/prompts_exemplos/<modelo>/... (arquivos .txt) e resumo no terminal.
"""
import argparse
import random
import re

from comum import (CATEGORIAS, K_KATE, MODELOS_PARA_TESTAR, PASTA_DADOS, SEMENTE,
                   IndiceSemantico, arquivo_resultados, carregar_folds, carregar_resultados,
                   montar_prompt, montar_prompt_probabilidades, normalizar, primeira_categoria,
                   remover_raciocinio, selecionar_fewshot)

PASTA_SAIDA = PASTA_DADOS / "prompts_exemplos"
METODOS = ["zeroshot", "fewshot", "kate", "ordenacao", "sg_icl", "ape", "globale", "locale", "calibracao"]
NOMES = {"zeroshot": "Zero-shot", "fewshot": "Few-shot", "kate": "KATE",
         "ordenacao": "Ordenação (KATE invertido)", "sg_icl": "SG-ICL", "ape": "APE",
         "globale": "GlobalE", "locale": "LocalE", "calibracao": "Calibração"}
GRUPO_DO_METODO = {"zeroshot": "basicos", "fewshot": "basicos", "kate": "basicos",
                   "ordenacao": "basicos", "sg_icl": "basicos", "ape": "avancados",
                   "globale": "avancados", "locale": "avancados", "calibracao": "calibracao"}
CHAVE_BRUTA = {"calibracao": "calibracao"}


class Contexto:
    """Carrega folds e resultados de um modelo e reconstrói os prompts."""

    def __init__(self, modelo):
        self.modelo = modelo
        self.folds = carregar_folds()
        self.resultados = {g: {f["fold"]: f for f in carregar_resultados(arquivo_resultados(g, modelo))}
                           for g in ("basicos", "avancados", "calibracao")}
        self.indices = {}

    def fold_da_pergunta(self, id_pergunta):
        for fold in self.folds:
            for item in fold["teste"]:
                if item["id"] == id_pergunta:
                    return fold, item
        raise SystemExit(f"Pergunta {id_pergunta} não encontrada nos folds.")

    def predicao(self, grupo, n_fold, id_pergunta):
        fold = self.resultados[grupo].get(n_fold)
        if not fold:
            return None, None
        for p in fold["predicoes"]:
            if p["id_pergunta"] == id_pergunta:
                return p, fold
        return None, fold

    def indice(self, fold):
        if fold["fold"] not in self.indices:
            self.indices[fold["fold"]] = IndiceSemantico(fold["treino"])
        return self.indices[fold["fold"]]

    def prompts(self, id_pergunta):
        fold, item = self.fold_da_pergunta(id_pergunta)
        n, pergunta, treino = fold["fold"], item["pergunta"], fold["treino"]
        por_id = {t["id"]: t for t in treino}
        demos_fewshot = selecionar_fewshot(treino, n)
        vizinhos, _ = self.indice(fold).vizinhos(pergunta, K_KATE)
        prompts = {
            "zeroshot": montar_prompt(pergunta),
            "fewshot": montar_prompt(pergunta, demos_fewshot),
            "kate": montar_prompt(pergunta, list(reversed(vizinhos))),
            "ordenacao": montar_prompt(pergunta, vizinhos),
            "calibracao": montar_prompt_probabilidades(pergunta, demos_fewshot),
        }
        # SG-ICL: exemplos gerados pelo LLM (lidos da resposta salva)
        p_bas, _ = self.predicao("basicos", n, id_pergunta)
        if p_bas:
            demos_sg = demonstracoes_sg_icl(p_bas["respostas_brutas"].get("sg_icl_geracao", ""), id_pergunta)
            prompts["sg_icl"] = montar_prompt(pergunta, demos_sg)
        # APE, GlobalE, LocalE: instrução e ordens escolhidas na validação
        _, f_av = self.predicao("avancados", n, id_pergunta)
        if f_av:
            prompts["ape"] = montar_prompt(pergunta, instrucao=f_av["ape"]["instrucao_escolhida"])
            aval = f_av["ordem"]["avaliacoes"]
            ordem_g = [por_id[i] for i in aval[f_av["ordem"]["indice_globale"]]["ordem_ids"]]
            ordem_l = [por_id[i] for i in aval[f_av["ordem"]["indice_locale"]]["ordem_ids"]]
            prompts["globale"] = montar_prompt(pergunta, ordem_g)
            prompts["locale"] = montar_prompt(pergunta, ordem_l)
        return fold, item, prompts


def demonstracoes_sg_icl(bruto, id_pergunta):
    """Mesma leitura usada no script 5 (etapa 1 do SG-ICL)."""
    demonstracoes, vistas = [], set()
    for linha in remover_raciocinio(bruto).splitlines():
        if "|" not in linha:
            continue
        rotulo, duvida = linha.split("|", 1)
        cat = primeira_categoria(normalizar(rotulo))
        duvida = re.sub(r"^[\s\-\d.)*]+", "", duvida).strip().strip('"*')
        if cat and cat not in vistas and len(duvida) > 5:
            demonstracoes.append({"pergunta": duvida, "categoria_tematica": cat})
            vistas.add(cat)
    random.Random(SEMENTE + id_pergunta).shuffle(demonstracoes)
    return demonstracoes


def relatorio_pergunta(ctx, id_pergunta, metodos=METODOS):
    fold, item, prompts = ctx.prompts(id_pergunta)
    linhas = [
        f"PERGUNTA {item['id']} | Fold {fold['fold']} | Modelo: {ctx.modelo}",
        f"Dúvida: {item['pergunta']}",
        f"Gabarito: {item['categoria_tematica']} | Tipologia: {item.get('tipologia', '')}",
        "",
    ]
    resumo = {}
    for m in metodos:
        p, _ = ctx.predicao(GRUPO_DO_METODO[m], fold["fold"], id_pergunta)
        previsto = p.get(f"previsto_{m}") if p else None
        bruto = p["respostas_brutas"].get(CHAVE_BRUTA.get(m, m), "") if p else ""
        resumo[m] = previsto
        status = "ainda não executado" if previsto is None else (
            "ACERTOU" if previsto == item["categoria_tematica"] else "ERROU")
        linhas += [
            "=" * 90,
            f"TÉCNICA: {NOMES[m]}  ->  previsto: {previsto}  [{status}]",
            "=" * 90,
            "----- PROMPT ENVIADO AO LLM -----",
            prompts.get(m, "(depende de resultados que ainda não foram gerados)"),
            "----- RESPOSTA BRUTA DO LLM -----",
            (bruto or "").strip() or "(sem resposta salva)",
            "",
        ]
    if "calibracao" in metodos:
        p, f = ctx.predicao("calibracao", fold["fold"], id_pergunta)
        if p and p.get("probabilidades"):
            linhas += ["Calibração — probabilidades p(y|x) e viés p_cf do fold:"]
            for c, px, pcf in zip(CATEGORIAS, p["probabilidades"], f["vies_p_cf"]):
                linhas.append(f"  {c:45s} p(y|x)={px:.3f}  p_cf={pcf:.3f}  p/p_cf={px / pcf:.2f}")
            linhas.append(f"  Sem correção: {p['previsto_calibracao_sem_correcao']} | "
                          f"Com correção: {p['previsto_calibracao']}")
    destino = PASTA_SAIDA / ctx.modelo
    destino.mkdir(parents=True, exist_ok=True)
    arquivo = destino / f"pergunta_{item['id']:03d}.txt"
    arquivo.write_text("\n".join(linhas), encoding="utf-8")
    return item, resumo, arquivo


def imprimir_resumo(itens):
    cab = f"{'id':>4} | {'gabarito':28s} | " + " | ".join(f"{NOMES[m][:9]:9s}" for m in METODOS)
    print(cab)
    print("-" * len(cab))
    for item, resumo in itens:
        celulas = []
        for m in METODOS:
            prev = resumo[m]
            celulas.append(f"{'—':9s}" if prev is None else
                           (f"{'ok':9s}" if prev == item["categoria_tematica"] else f"{prev[:9]:9s}"))
        print(f"{item['id']:>4} | {item['categoria_tematica'][:28]:28s} | " + " | ".join(celulas))
    print("\n'ok' = acertou | texto = categoria prevista (errada) | — = ainda não executado")


def main():
    parser = argparse.ArgumentParser(description="Exibe os prompts finais de cada técnica.")
    parser.add_argument("--modelo", required=True, choices=MODELOS_PARA_TESTAR)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--id", type=int, nargs="+", help="id(s) da(s) pergunta(s)")
    grupo.add_argument("--uma-por-categoria", action="store_true",
                       help="1 pergunta de teste de cada categoria (fold 1)")
    grupo.add_argument("--erros", choices=METODOS, help="casos em que esta técnica errou")
    parser.add_argument("--n", type=int, default=5, help="nº de casos para --erros")
    args = parser.parse_args()

    ctx = Contexto(args.modelo)
    if args.id:
        ids = args.id
    elif args.uma_por_categoria:
        teste = ctx.folds[0]["teste"]
        ids = [next((t["id"] for t in teste if t["categoria_tematica"] == c), None) for c in CATEGORIAS]
        ids = [i for i in ids if i is not None]
    else:
        grupo_m = GRUPO_DO_METODO[args.erros]
        erros = [p["id_pergunta"] for f in ctx.resultados[grupo_m].values() for p in f["predicoes"]
                 if p[f"previsto_{args.erros}"] != p["gabarito"]]
        if not erros:
            raise SystemExit(f"Nenhum erro encontrado (ou {args.erros} ainda não foi executado).")
        ids = random.Random(SEMENTE).sample(erros, min(args.n, len(erros)))

    itens = []
    for id_pergunta in ids:
        item, resumo, arquivo = relatorio_pergunta(ctx, id_pergunta)
        itens.append((item, resumo))
        print(f"Prompt salvo: {arquivo}")
    print()
    imprimir_resumo(itens)


if __name__ == "__main__":
    main()
