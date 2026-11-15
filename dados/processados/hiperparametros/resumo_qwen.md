## Ajuste de hiperparâmetros na validação · qwen

F1-macro (%) na validação (100 perguntas por fold), média nos folds 1, 2, 3, 4, 5.
Em negrito: o valor fixado nos experimentos principais. p = teste t pareado contra o valor fixado.

| Configuração | F1-macro validação | Acurácia validação | p vs. valor fixado |
|---|---|---|---|
| KATE k=1 | 85,70 | 85,00 | 0,488 |
| KATE k=3 | 86,47 | 86,00 | 0,994 |
| **KATE k=5** | 86,48 | 87,40 | — |
| KATE k=8 | 88,23 | 88,40 | 0,093 |
| KATE k=10 | 87,89 | 87,80 | 0,135 |
| **Few-shot 1/categoria (11 ex.)** | 84,83 | 84,60 | — |
| Few-shot 2/categoria (22 ex.) | 87,37 | 87,00 | 0,043 |
| Few-shot 3/categoria (33 ex.) | 87,73 | 86,40 | 0,032 |

### Teste com o valor escolhido na validação

| Fold | Técnica | Valor escolhido | F1 teste (escolhido) | F1 teste (fixado) |
|---|---|---|---|---|
| 1 | kate | 10 | 81,05 | 79,36 |
| 1 | fewshot | 2 | 79,30 | 82,54 |
| 2 | kate | 5 | 80,87 | 80,87 |
| 2 | fewshot | 3 | 78,81 | 76,31 |
| 3 | kate | 8 | 86,95 | 87,05 |
| 3 | fewshot | 3 | 76,43 | 83,15 |
| 4 | kate | 8 | 75,90 | 77,47 |
| 4 | fewshot | 3 | 75,25 | 73,60 |
| 5 | kate | 8 | 90,40 | 91,09 |
| 5 | fewshot | 2 | 89,34 | 89,99 |
