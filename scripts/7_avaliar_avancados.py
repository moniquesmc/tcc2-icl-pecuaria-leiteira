"""
Técnicas 6 a 8: APE, GlobalE e LocalE.

Uso:
    python scripts/7_avaliar_avancados.py                  # os 4 modelos
    python scripts/7_avaliar_avancados.py --modelo gemini  # só um modelo

Estas técnicas têm uma ETAPA DE SELEÇÃO feita no conjunto de VALIDAÇÃO de cada fold
(nunca no teste), seguida da avaliação no conjunto de teste.

- APE (Zhou et al., 2022): o LLM gera instruções candidatas a partir de pares
  entrada-saída do treino (modo "forward generation"); cada candidata é pontuada pela
  acurácia de execução na validação; a melhor é usada no teste (zero-shot).
- GlobalE e LocalE (Lu et al., 2022): partindo das MESMAS 11 demonstrações do Few-shot,
  testam várias ordens (permutações). Para cada ordem, o modelo classifica um conjunto de
  sondagem (perguntas da validação, SEM usar o gabarito):
    * GlobalE = entropia da distribuição dos rótulos previstos no conjunto de sondagem;
    * LocalE  = média da entropia da distribuição de probabilidades de cada pergunta.
  Escolhe-se a ordem de MAIOR entropia (menos enviesada) e ela é usada no teste.
  Adaptação: como as APIs não expõem as probabilidades dos tokens, as probabilidades
  são verbalizadas pelo modelo (formato '<categoria>: X%').
"""
import random

from comum import (CATEGORIAS, N_CANDIDATOS_APE, N_PERMUTACOES, SEMENTE, amostra_validacao,
                   chamar_com_retry, entropia, executar_experimento, extrair_classe_prevista,
                   extrair_distribuicao, formatar_demonstracoes, montar_prompt,
                   montar_prompt_probabilidades, progresso, remover_raciocinio,
                   selecionar_fewshot)

GRUPO = "avancados"


# ======================= APE =======================
def gerar_instrucao_candidata(demonstracoes, modelo):
    prompt = (
        "Eu dei uma instrução a um amigo. Seguindo essa instrução, ele leu cada dúvida de "
        "produtor rural abaixo e escreveu a categoria técnica correspondente. "
        "Estes são os pares de entrada e saída:\n\n"
        f"{formatar_demonstracoes(demonstracoes)}"
        "Qual foi a instrução que eu dei ao meu amigo? Responda apenas com a instrução, "
        "em português, em um único parágrafo."
    )
    bruto = chamar_com_retry(prompt, modelo)
    instrucao = remover_raciocinio(bruto).strip().strip('"*')
    for prefixo in ("Instrução:", "Instrucao:", "A instrução foi:", "A instrução era:"):
        if instrucao.lower().startswith(prefixo.lower()):
            instrucao = instrucao[len(prefixo):].strip()
    return instrucao[:1200], bruto


def selecionar_instrucao_ape(treino, validacao_amostra, n_fold, modelo):
    rng = random.Random(SEMENTE + 2000 + n_fold)
    por_classe = {c: [t for t in treino if t["categoria_tematica"] == c] for c in CATEGORIAS}
    candidatas = []
    for c in range(N_CANDIDATOS_APE):
        # cada candidata é gerada a partir de um conjunto diferente de demonstrações (1 por classe)
        demos = [rng.choice(por_classe[cat]) for cat in CATEGORIAS if por_classe[cat]]
        rng.shuffle(demos)
        instrucao, bruto = gerar_instrucao_candidata(demos, modelo)
        if not instrucao:
            continue
        acertos = 0
        for j, item in enumerate(validacao_amostra, 1):
            progresso(modelo, n_fold, j, len(validacao_amostra), f"APE candidata {c + 1}: validação ")
            resposta = chamar_com_retry(montar_prompt(item["pergunta"], instrucao=instrucao), modelo)
            acertos += extrair_classe_prevista(resposta) == item["categoria_tematica"]
        acuracia = acertos / len(validacao_amostra)
        candidatas.append({"instrucao": instrucao, "acuracia_validacao": acuracia,
                           "resposta_bruta_geracao": bruto})
        print(f"\n  APE candidata {c + 1}: acurácia na validação = {acuracia:.2%}")
    if not candidatas:
        raise RuntimeError("O modelo não gerou nenhuma instrução válida para o APE.")
    melhor = max(candidatas, key=lambda x: x["acuracia_validacao"])   # empate: a primeira
    return melhor["instrucao"], candidatas


# ======================= GlobalE / LocalE =======================
def gerar_permutacoes(demonstracoes, n_fold):
    rng = random.Random(SEMENTE + 3000 + n_fold)
    permutacoes = [list(demonstracoes)]          # a ordem original do Few-shot entra na disputa
    tentativas = 0
    while len(permutacoes) < N_PERMUTACOES and tentativas < 1000:
        p = list(demonstracoes)
        rng.shuffle(p)
        if [d["id"] for d in p] not in [[d["id"] for d in q] for q in permutacoes]:
            permutacoes.append(p)
        tentativas += 1
    return permutacoes


def selecionar_ordem_entropia(demos_fewshot, sondagem, n_fold, modelo):
    avaliacoes = []
    for k, perm in enumerate(gerar_permutacoes(demos_fewshot, n_fold)):
        rotulos, entropias_locais, n_ilegiveis = [], [], 0
        for j, item in enumerate(sondagem, 1):
            progresso(modelo, n_fold, j, len(sondagem), f"GlobalE/LocalE ordem {k + 1}: sondagem ")
            resposta = chamar_com_retry(montar_prompt_probabilidades(item["pergunta"], perm), modelo)
            dist = extrair_distribuicao(resposta)
            if dist is None:
                n_ilegiveis += 1
                rotulo = extrair_classe_prevista(resposta)
                if rotulo in CATEGORIAS:          # sem probabilidades: conta só o rótulo (entropia local 0)
                    rotulos.append(rotulo)
                    entropias_locais.append(0.0)
                continue
            rotulos.append(CATEGORIAS[int(dist.argmax())])
            entropias_locais.append(entropia(dist))
        contagem = [rotulos.count(c) / len(rotulos) for c in CATEGORIAS] if rotulos else [0]
        avaliacoes.append({
            "ordem_ids": [d["id"] for d in perm],
            "globale": entropia(contagem),
            "locale": sum(entropias_locais) / len(entropias_locais) if entropias_locais else 0.0,
            "respostas_ilegiveis": n_ilegiveis,
        })
        print(f"\n  Ordem {k + 1}: GlobalE={avaliacoes[-1]['globale']:.3f} "
              f"LocalE={avaliacoes[-1]['locale']:.3f}")
    melhor_g = max(range(len(avaliacoes)), key=lambda i: avaliacoes[i]["globale"])
    melhor_l = max(range(len(avaliacoes)), key=lambda i: avaliacoes[i]["locale"])
    por_id = {d["id"]: d for d in demos_fewshot}
    ordem_g = [por_id[i] for i in avaliacoes[melhor_g]["ordem_ids"]]
    ordem_l = [por_id[i] for i in avaliacoes[melhor_l]["ordem_ids"]]
    return ordem_g, ordem_l, avaliacoes, melhor_g, melhor_l


# ======================= FOLD =======================
def processar_fold(fold, modelo):
    treino, validacao, teste, n_fold = fold["treino"], fold["validacao"], fold["teste"], fold["fold"]
    amostra_val = amostra_validacao(validacao, n_fold)
    demos_fewshot = selecionar_fewshot(treino, n_fold)

    print("Etapa de seleção do APE (validação)...")
    instrucao_ape, candidatas_ape = selecionar_instrucao_ape(treino, amostra_val, n_fold, modelo)

    print("Etapa de seleção da ordem - GlobalE/LocalE (sondagem na validação)...")
    ordem_g, ordem_l, avaliacoes, idx_g, idx_l = selecionar_ordem_entropia(
        demos_fewshot, amostra_val, n_fold, modelo)

    print("Avaliação no conjunto de teste...")
    predicoes = []
    for i, item in enumerate(teste, 1):
        progresso(modelo, n_fold, i, len(teste), "teste ")
        pergunta = item["pergunta"]
        bruto_ape = chamar_com_retry(montar_prompt(pergunta, instrucao=instrucao_ape), modelo)
        bruto_g = chamar_com_retry(montar_prompt(pergunta, ordem_g), modelo)
        bruto_l = bruto_g if idx_l == idx_g else chamar_com_retry(montar_prompt(pergunta, ordem_l), modelo)
        predicoes.append({
            "id_pergunta": item["id"],
            "gabarito": item["categoria_tematica"],
            "previsto_ape": extrair_classe_prevista(bruto_ape),
            "previsto_globale": extrair_classe_prevista(bruto_g),
            "previsto_locale": extrair_classe_prevista(bruto_l),
            "respostas_brutas": {"ape": bruto_ape, "globale": bruto_g, "locale": bruto_l},
        })

    return {
        "fold": n_fold,
        "validacao_ids_usados": [v["id"] for v in amostra_val],
        "ape": {"instrucao_escolhida": instrucao_ape, "candidatas": candidatas_ape},
        "ordem": {"avaliacoes": avaliacoes, "indice_globale": idx_g, "indice_locale": idx_l},
        "predicoes": predicoes,
    }


if __name__ == "__main__":
    executar_experimento(GRUPO, "TÉCNICAS 6-8 (APE, GlobalE, LocalE)", processar_fold)
