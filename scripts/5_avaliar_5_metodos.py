"""
Técnicas 1 a 5: Zero-shot, Few-shot, KATE, Ordenação por similaridade e SG-ICL.

Uso:
    python scripts/5_avaliar_5_metodos.py                  # os 4 modelos
    python scripts/5_avaliar_5_metodos.py --modelo gemini  # só um modelo

Referências:
- Zero-shot / Few-shot: Brown et al. (2020).
- KATE: Liu et al. (2022) - k vizinhos mais similares (embeddings), com o MAIS similar
  posicionado mais perto da pergunta (último exemplo).
- Ordenação por similaridade: mesmos k exemplos do KATE em ordem inversa (o mais similar
  primeiro), para medir o efeito da ordem das demonstrações (viés de recência).
- SG-ICL: Kim et al. (2022) - o próprio LLM gera demonstrações condicionadas à pergunta de
  teste e a cada rótulo (etapa 1) e depois classifica usando-as (etapa 2).
"""
import random
import re

from comum import (CATEGORIAS, K_KATE, SEMENTE, IndiceSemantico, chamar_com_retry,
                   executar_experimento, extrair_classe_prevista, lista_categorias,
                   montar_prompt, normalizar, primeira_categoria, progresso,
                   remover_raciocinio, selecionar_fewshot)

GRUPO = "basicos"


def gerar_demonstracoes_sg_icl(pergunta, modelo, id_pergunta):
    """Etapa 1 do SG-ICL: gera 1 dúvida nova por categoria, inspirada na dúvida de teste."""
    prompt = (
        "Você é especialista em pecuária leiteira. Veja a dúvida de um produtor rural:\n"
        f"Dúvida: {pergunta}\n\n"
        "Para CADA uma das 11 categorias abaixo, escreva UMA nova dúvida curta de produtor rural, "
        "no mesmo estilo da dúvida acima, mas que pertença claramente àquela categoria.\n"
        f"Categorias: {lista_categorias()}.\n\n"
        "Responda exatamente em 11 linhas, sem explicações, no formato:\n"
        "<nome da categoria> | <dúvida gerada>"
    )
    bruto = chamar_com_retry(prompt, modelo)
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
    # ordem aleatória (semente fixa) para não favorecer a última categoria da lista
    random.Random(SEMENTE + id_pergunta).shuffle(demonstracoes)
    return demonstracoes, bruto


def processar_fold(fold, modelo):
    treino, teste, n_fold = fold["treino"], fold["teste"], fold["fold"]
    demos_fewshot = selecionar_fewshot(treino, n_fold)
    indice = IndiceSemantico(treino)

    predicoes = []
    for i, item in enumerate(teste, 1):
        progresso(modelo, n_fold, i, len(teste), "pergunta ")
        pergunta = item["pergunta"]

        vizinhos, similaridades = indice.vizinhos(pergunta, K_KATE)   # mais -> menos similar
        demos_kate = list(reversed(vizinhos))      # KATE: mais similar por último (perto da pergunta)
        demos_ordenacao = vizinhos                 # Ordenação: mais similar primeiro

        demos_sg, bruto_sg_geracao = gerar_demonstracoes_sg_icl(pergunta, modelo, item["id"])

        prompts = {
            "zeroshot": montar_prompt(pergunta),
            "fewshot": montar_prompt(pergunta, demos_fewshot),
            "kate": montar_prompt(pergunta, demos_kate),
            "ordenacao": montar_prompt(pergunta, demos_ordenacao),
            "sg_icl": montar_prompt(pergunta, demos_sg),
        }
        brutos = {m: chamar_com_retry(p, modelo) for m, p in prompts.items()}

        registro = {"id_pergunta": item["id"], "gabarito": item["categoria_tematica"]}
        registro.update({f"previsto_{m}": extrair_classe_prevista(b) for m, b in brutos.items()})
        registro["respostas_brutas"] = {**brutos, "sg_icl_geracao": bruto_sg_geracao}
        registro["detalhes"] = {
            "kate_ids": [d["id"] for d in demos_kate],
            "kate_similaridades": [round(s, 4) for s in reversed(similaridades)],
            "sg_icl_n_demonstracoes": len(demos_sg),
        }
        predicoes.append(registro)

    return {
        "fold": n_fold,
        "fewshot_ids": [d["id"] for d in demos_fewshot],
        "predicoes": predicoes,
    }


if __name__ == "__main__":
    executar_experimento(GRUPO, "TÉCNICAS 1-5 (Zero-shot, Few-shot, KATE, Ordenação, SG-ICL)",
                         processar_fold)
