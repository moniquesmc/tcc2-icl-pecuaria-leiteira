# Estudo comparativo de técnicas de In-Context Learning em LLMs aplicado à pecuária leiteira

Código do Trabalho de Conclusão de Curso de **Monique Silva Machado** (Bacharelado em Ciência da Computação, IFG Câmpus Anápolis), orientado pelo Prof. Dr. Sergio Daniel Carvalho Canuto.

O trabalho compara **9 técnicas de In-Context Learning** na classificação das 500 perguntas do manual *500 Perguntas 500 Respostas – Gado de Leite* (Embrapa) nas **11 categorias** correspondentes aos capítulos do livro, em **4 modelos de linguagem**.

| Frente (Dong et al., 2024) | Técnicas |
|---|---|
| Linhas de base | Zero-shot, Few-shot |
| Demonstração: seleção | KATE, SG-ICL |
| Demonstração: ordem | Ordenação (KATE invertido), GlobalE, LocalE |
| Instrução | APE |
| Função de pontuação | Calibração contextual |

Modelos: Llama 3.2 3B (local, Ollama), Qwen3.8 27B, Llama 3.3 70B e Gemini 3.1 Flash-Lite (via OpenRouter). Temperatura 0 em todos.

## Protocolo

- Validação cruzada estratificada em 5 *folds* (semente 42): 300 perguntas de treino, 100 de validação e 100 de teste por *fold*.
- Métricas: acurácia, precisão, recall e F1-macro (média ± IC 95%).
- Teste t pareado entre a melhor técnica de cada modelo e as demais (p > 0,05: sem diferença estatisticamente significativa).
- Todas as respostas dos modelos ficam em `dados/processados/cache_respostas/`, o que permite reproduzir as métricas sem novas chamadas.

## Estrutura

```
scripts/
  1_extrair_pdf.py                    extrai o texto do PDF do manual
  2_extrair_perguntas_com_categorias.py  separa as 500 perguntas e a categoria (capítulo)
  3_classificar_tipologia.py, 3b_revisar_tipologia.py  tipologia (booleana/descritiva)
  4_preparar_cv.py                    gera os 5 folds
  5_avaliar_5_metodos.py              Zero-shot, Few-shot, KATE, Ordenação, SG-ICL
  7_avaliar_avancados.py              APE, GlobalE, LocalE
  8_avaliar_calibracao.py             Calibração contextual
  6_calcular_metricas.py              tabelas, testes t e gráficos
  9_exibir_prompts.py                 reconstrói o prompt de cada técnica (análise de erros)
  10_gerar_apresentacao.py            apresentação HTML interativa com os resultados
  11_ajustar_hiperparametros.py       testa k do KATE e nº de exemplos do Few-shot na validação
  contar_categorias.py, teste_*.py    utilitários (contagem por categoria, teste das APIs)
  comum.py                            configurações, prompts e funções compartilhadas
dados/processados/                    perguntas estruturadas, folds, resultados, gráficos e cache
apresentacao/                         modelo da apresentação
```

## Como reproduzir

```bash
python -m venv venv
venv\Scripts\activate          # Windows  (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # e preencha as chaves de API
```

1. Baixe o PDF do manual no site da Embrapa e salve em `dados/brutos/500perguntasgadoleite.pdf` (o livro não é distribuído neste repositório).
2. Rode os scripts na ordem: 1, 2, 3, 3b, 4, 5, 7, 8 e 6. Use `--modelo ollama|qwen|llama-70b|gemini` para escolher o modelo.
3. Para apenas recalcular as métricas com as respostas já salvas, basta rodar `python scripts/6_calcular_metricas.py` e `python scripts/10_gerar_apresentacao.py`.
4. Ajuste de hiperparâmetros (opcional): `python scripts/11_ajustar_hiperparametros.py --modelo qwen --avaliar-teste`.

## Principais resultados (F1-macro, média dos 5 folds)

| Modelo | Melhor técnica | F1-macro |
|---|---|---|
| Llama 3.2 3B | Ordenação | 36,6% |
| Qwen3.8 27B | KATE | 83,2% |
| Llama 3.3 70B | KATE | 77,4% |
| Gemini 3.1 Flash-Lite | LocalE | 81,0% |

Tabelas completas em `dados/processados/tabela_f1_macro.md` e `dados/processados/comparativo_9_tecnicas.csv`.
