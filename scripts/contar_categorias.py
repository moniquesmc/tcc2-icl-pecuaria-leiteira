import json
from collections import Counter
from pathlib import Path

DIRETORIO_ATUAL = Path(__file__).resolve().parent
ARQUIVO_PERGUNTAS = DIRETORIO_ATUAL.parent / "dados" / "processados" / "perguntas_estruturadas_final.json"

# As 11 categorias oficiais do TCC
CATEGORIAS = [
    "Cria e Recria de Bezerras e Novilhas", "Alimentação e Manejo de Vacas e Touros",
    "Recursos Forrageiros", "Reprodução", "Melhoramento Genético Animal",
    "Saúde Animal", "Mastite e Qualidade do Leite", "Produção Orgânica de Leite",
    "Instalações, Ambiência e Manejo de Dejetos", "Bem-estar Animal",
    "Gerenciamento da Atividade Leiteira"
]

def main():
    if not ARQUIVO_PERGUNTAS.exists():
        print(f"Erro: O arquivo {ARQUIVO_PERGUNTAS} não foi encontrado.")
        return

    print("A carregar base de dados...\n")
    with open(ARQUIVO_PERGUNTAS, 'r', encoding='utf-8') as f:
        perguntas = json.load(f)

    # Conta padronizando tudo para minúsculas (evita erros se houver diferenças no JSON)
    contador = Counter(item['categoria_tematica'].strip().lower() for item in perguntas)

    print(f"Total geral de perguntas no JSON: {len(perguntas)}")
    print("=" * 70)
    print(f"{'QUANTIDADE':<12} | {'CATEGORIA'}")
    print("=" * 70)
    
    # Cruza a contagem com a lista oficial de 11 categorias
    resultados = []
    for cat in CATEGORIAS:
        qtd = contador.get(cat.lower(), 0)
        resultados.append((cat, qtd))
        
    # Ordena da categoria com mais perguntas para a que tem menos
    resultados.sort(key=lambda x: x[1], reverse=True)
    
    for cat, qtd in resultados:
        print(f"{qtd:<12} | {cat}")
        
    print("=" * 70)

if __name__ == "__main__":
    main()