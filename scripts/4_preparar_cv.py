import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

DIRETORIO_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = DIRETORIO_ATUAL.parent / "dados" / "processados" / "perguntas_estruturadas_final.json"
ARQUIVO_SAIDA = DIRETORIO_ATUAL.parent / "dados" / "processados" / "folds_cv.json"

def main():
    print("Carregando a base de dados final...")
    with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # Prepara os dados para o Scikit-Learn
    # Extrai as categorias para servir de base para a estratificação
    categorias = [item["categoria_tematica"] for item in dados]

    print("Gerando 5-Fold Cross Validation (Estratificado, Seed=42)...")
    # Cria o gerador de K-Folds (Test = 20%)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    folds = []

    # O skf.split divide em 80% (Treino+Validação) e 20% (Teste)
    for fold_idx, (treino_val_indices, teste_indices) in enumerate(skf.split(dados, categorias)):
        
        # Pega as categorias apenas dos 80% para fazer a segunda quebra (Treino / Validação)
        categorias_treino_val = [categorias[i] for i in treino_val_indices]
        
        # Dos 80%, quebramos novamente: 25% para validação e 75% para treino
        # Isso equivale exatamente a 20% do total para Validação e 60% do total para Treino!
        treino_idx, val_idx = train_test_split(
            treino_val_indices, 
            test_size=0.25, 
            stratify=categorias_treino_val, 
            random_state=42
        )

        # Monta as listas de objetos completos para cada conjunto
        fold_treino = [dados[i] for i in treino_idx]
        fold_val = [dados[i] for i in val_idx]
        fold_teste = [dados[i] for i in teste_indices]

        folds.append({
            "fold": fold_idx + 1,
            "treino": fold_treino,       # ~300 itens
            "validacao": fold_val,       # ~100 itens
            "teste": fold_teste          # ~100 itens
        })

        print(f"Fold {fold_idx + 1} criado: Treino={len(fold_treino)} | Val={len(fold_val)} | Teste={len(fold_teste)}")

    # Salva a estrutura de Folds num arquivo novo
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)

    print(f"\nSucesso! Arquivo de Validação Cruzada salvo em:\n{ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()