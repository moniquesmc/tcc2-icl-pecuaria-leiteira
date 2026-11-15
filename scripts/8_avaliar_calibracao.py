"""
Técnica 9: Calibração Contextual (Zhao et al., 2021).

Uso:
    python scripts/8_avaliar_calibracao.py                  # os 4 modelos
    python scripts/8_avaliar_calibracao.py --modelo gemini  # só um modelo

Procedimento (por fold), usando as MESMAS 11 demonstrações do Few-shot:
1. Mede o viés do modelo: envia o prompt com entradas SEM conteúdo ("N/A", "" e "[MASK]")
   e obtém a distribuição de probabilidades das 11 categorias; p_cf = média normalizada.
2. Para cada pergunta de teste, obtém a distribuição p(y|x) com o mesmo prompt.
3. Corrige o viés: q(y|x) ∝ p(y|x) / p_cf(y)   (W = diag(p_cf)^-1, b = 0)
   e a categoria prevista é argmax q.
Adaptação: como as APIs não expõem as probabilidades dos tokens, as probabilidades são
verbalizadas pelo modelo (formato '<categoria>: X%'). Aplica-se suavização (epsilon)
para evitar divisão por zero.
Também é salva a previsão SEM a correção (argmax p), para medir o efeito da calibração.
"""
import numpy as np

from comum import (CATEGORIAS, ENTRADAS_NULAS, chamar_com_retry, executar_experimento,
                   extrair_classe_prevista, extrair_distribuicao, montar_prompt_probabilidades,
                   progresso, selecionar_fewshot)

GRUPO = "calibracao"
SUAVIZACAO = 0.01


def estimar_vies(demos, modelo):
    distribuicoes, brutos = [], {}
    for entrada in ENTRADAS_NULAS:
        resposta = chamar_com_retry(montar_prompt_probabilidades(entrada, demos), modelo)
        brutos[entrada or "<vazio>"] = resposta
        dist = extrair_distribuicao(resposta, suavizacao=SUAVIZACAO)
        if dist is not None:
            distribuicoes.append(dist)
    if not distribuicoes:
        print("  [aviso] Não foi possível ler o viés; usando distribuição uniforme (sem correção).")
        return np.full(len(CATEGORIAS), 1 / len(CATEGORIAS)), brutos, 0
    p_cf = np.mean(distribuicoes, axis=0)
    return p_cf / p_cf.sum(), brutos, len(distribuicoes)


def processar_fold(fold, modelo):
    treino, teste, n_fold = fold["treino"], fold["teste"], fold["fold"]
    demos = selecionar_fewshot(treino, n_fold)

    print("Estimando o viés do modelo com entradas sem conteúdo...")
    p_cf, brutos_cf, n_validas = estimar_vies(demos, modelo)
    print("  Viés p_cf: " + ", ".join(f"{c[:12]}={p:.2f}" for c, p in zip(CATEGORIAS, p_cf)))

    predicoes = []
    for i, item in enumerate(teste, 1):
        progresso(modelo, n_fold, i, len(teste), "pergunta ")
        resposta = chamar_com_retry(montar_prompt_probabilidades(item["pergunta"], demos), modelo)
        dist = extrair_distribuicao(resposta, suavizacao=SUAVIZACAO)
        if dist is None:
            # sem probabilidades legíveis: não há como calibrar -> usa o rótulo escrito
            # (fica marcado em "calibrado": False para ser reportado na análise)
            sem_correcao = extrair_classe_prevista(resposta)
            calibrada = sem_correcao
        else:
            sem_correcao = CATEGORIAS[int(dist.argmax())]
            q = dist / p_cf
            calibrada = CATEGORIAS[int(q.argmax())]
        predicoes.append({
            "id_pergunta": item["id"],
            "gabarito": item["categoria_tematica"],
            "previsto_calibracao": calibrada,
            "previsto_calibracao_sem_correcao": sem_correcao,
            "calibrado": dist is not None,
            "probabilidades": None if dist is None else [round(float(p), 4) for p in dist],
            "respostas_brutas": {"calibracao": resposta},
        })

    return {
        "fold": n_fold,
        "fewshot_ids": [d["id"] for d in demos],
        "vies_p_cf": [round(float(p), 4) for p in p_cf],
        "vies_entradas_validas": n_validas,
        "vies_respostas_brutas": brutos_cf,
        "predicoes": predicoes,
    }


if __name__ == "__main__":
    executar_experimento(GRUPO, "TÉCNICA 9 (Calibração Contextual)", processar_fold)
