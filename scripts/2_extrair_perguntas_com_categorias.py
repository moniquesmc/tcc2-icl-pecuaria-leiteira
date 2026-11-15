import json
import re
from pathlib import Path

DIRETORIO_ATUAL = Path(__file__).resolve().parent
ARQUIVO_PAGINAS = DIRETORIO_ATUAL.parent / "dados" / "processados" / "texto_bruto_paginas.json"
ARQUIVO_SAIDA = DIRETORIO_ATUAL.parent / "dados" / "processados" / "perguntas_estruturadas_parcial.json"

# Primeira pergunta de cada capítulo do livro "500 Perguntas 500 Respostas - Gado de Leite"
INICIO_CAPITULOS = [
    (1,   "Cria e Recria de Bezerras e Novilhas"),          # Cap. 1
    (41,  "Alimentação e Manejo de Vacas e Touros"),        # Cap. 2
    (135, "Recursos Forrageiros"),                          # Cap. 3
    (290, "Reprodução"),                                    # Cap. 4
    (340, "Melhoramento Genético Animal"),                  # Cap. 5
    (380, "Saúde Animal"),                                  # Cap. 6
    (415, "Mastite e Qualidade do Leite"),                  # Cap. 7
    (450, "Produção Orgânica de Leite"),                    # Cap. 8
    (458, "Gerenciamento da Atividade Leiteira"),           # Cap. 9
    (469, "Bem-estar Animal"),                              # Cap. 10
    (483, "Instalações, Ambiência e Manejo de Dejetos"),    # Cap. 11
]

def categoria_por_id(id_pergunta):
    categoria = INICIO_CAPITULOS[0][1]
    for inicio, nome in INICIO_CAPITULOS:
        if id_pergunta >= inicio:
            categoria = nome
    return categoria

def main():
    with open(ARQUIVO_PAGINAS, 'r', encoding='utf-8') as f:
        paginas = json.load(f)

    print("Limpando e reprocessando o texto com precisão...")
    
    # 1. Junta as páginas e remove quebras de linha
    texto_bruto = " ".join([pag.replace('\n', ' ') for pag in paginas])
    
    # 2. CORTA A INTRODUÇÃO
    marcador_inicio = "1 Quando iniciar os cuidados com os bezerros?"
    idx_inicio = texto_bruto.find(marcador_inicio)
    if idx_inicio != -1:
        texto_bruto = texto_bruto[idx_inicio:]

    posicoes = []
    
    # 3. Encontra onde cada pergunta começa.
    # A busca da pergunta i começa DEPOIS da pergunta i-1: assim números de página
    # ou quantidades no texto (ex.: "21", "150 L/animal/dia") não são confundidos
    # com o número da pergunta (erro que afetava as perguntas 21 e 150).
    pos_atual = 0
    for i in range(1, 501):
        padrao = re.compile(r'(?<![\d/.,])\b' + str(i) + r'\s+([A-ZÉÓÍÁÂÊÔÃÕÇ][^\?]{5,300}\?)')
        match = padrao.search(texto_bruto, pos_atual)
        if match:
            pos_atual = match.end()
        
        if match:
            posicoes.append({
                "id": i,
                "inicio": match.start(),
                "fim_pergunta": match.end(),
                "pergunta": match.group(1).strip()
            })

    posicoes.sort(key=lambda x: x["inicio"])

    dados_json = []

    for idx, pos in enumerate(posicoes):
        inicio_resp = pos["fim_pergunta"]
        fim_resp = posicoes[idx+1]["inicio"] if idx + 1 < len(posicoes) else len(texto_bruto)
        
        resposta_crua = texto_bruto[inicio_resp:fim_resp]
        
        # =====================================================================
        # 4. MAPEAMENTO EXATO DE CAPÍTULOS (conferido no sumário do PDF da Embrapa)
        # A categoria é definida pelo NÚMERO da pergunta, não pela posição no texto.
        # =====================================================================
        categoria_atual = categoria_por_id(pos["id"])

        # 5. LIMPEZA FINAL DA RESPOSTA
        resposta_limpa = re.sub(r'\s+', ' ', resposta_crua).strip()
        resposta_limpa = re.sub(r'\s+\d{1,3}$', '', resposta_limpa).strip() 
        
        dados_json.append({
            "id": pos["id"],
            "pergunta": pos["pergunta"],
            "resposta": resposta_limpa,
            "categoria_tematica": categoria_atual,
            "tipologia": ""
        })

    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(dados_json, f, ensure_ascii=False, indent=2)

    print(f"Corrigido! {len(dados_json)} perguntas salvas, limpas e categorizadas.")

if __name__ == "__main__":
    main()