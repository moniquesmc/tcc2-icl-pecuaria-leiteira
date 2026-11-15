"""
Teste rápido: os modelos respondem as probabilidades no formato que o código entende?
(usado por GlobalE, LocalE e Calibração). Faz 1 chamada por modelo; a resposta fica no
cache e é reaproveitada depois pelo script 8.

Uso:
    python scripts/teste_probabilidades.py
"""
from comum import (CATEGORIAS, MODELOS_PARA_TESTAR, carregar_folds, chamar_com_retry,
                   extrair_distribuicao, montar_prompt_probabilidades, selecionar_fewshot)


def main():
    fold = carregar_folds()[0]
    demos = selecionar_fewshot(fold["treino"], fold["fold"])
    item = fold["teste"][0]
    prompt = montar_prompt_probabilidades(item["pergunta"], demos)
    print(f"Pergunta: {item['pergunta']}\nGabarito: {item['categoria_tematica']}\n")
    for modelo in MODELOS_PARA_TESTAR:
        if modelo == "llama-nvidia":
            continue
        print("=" * 70, f"\n{modelo.upper()}")
        try:
            resposta = chamar_com_retry(prompt, modelo, max_tentativas=2)
        except Exception as e:
            print(f"  ERRO: {e}")
            continue
        print("  Resposta bruta:\n   " + resposta.strip().replace("\n", "\n   ")[:900])
        dist = extrair_distribuicao(resposta)
        if dist is None:
            print("  >>> PROBLEMA: não foi possível ler as probabilidades.")
            continue
        lidas = sum(1 for p in dist if p > 0)
        maior = CATEGORIAS[int(dist.argmax())]
        print(f"  >>> OK: {lidas} categorias com probabilidade > 0; mais provável: {maior}")


if __name__ == "__main__":
    main()
