import json
from pathlib import Path

# Tenta importar o pypdf. Se não tiver, avisa o usuário.
try:
    from pypdf import PdfReader
except ImportError:
    print("Biblioteca 'pypdf' não encontrada.")
    print("Por favor, abra o terminal e rode: pip install pypdf")
    exit()

# Mapeia os caminhos absolutos baseados na localização deste script
DIRETORIO_ATUAL = Path(__file__).resolve().parent
ARQUIVO_PDF = DIRETORIO_ATUAL.parent / "dados" / "brutos" / "500perguntasgadoleite.pdf"
ARQUIVO_SAIDA = DIRETORIO_ATUAL.parent / "dados" / "processados" / "texto_bruto_paginas.json"

def main():
    if not ARQUIVO_PDF.exists():
        print(f"Erro: O arquivo PDF não foi encontrado exatamente em:\n{ARQUIVO_PDF}")
        return

    print("Iniciando a leitura do PDF da Embrapa...")
    reader = PdfReader(ARQUIVO_PDF)
    
    paginas_texto = []
    
    # Extrai o texto de cada página sequencialmente
    for i, pagina in enumerate(reader.pages):
        texto = pagina.extract_text()
        if texto:
            paginas_texto.append(texto)
            
    # Salva o resultado no formato JSON na pasta processados
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(paginas_texto, f, ensure_ascii=False, indent=2)

    print(f"Sucesso! {len(paginas_texto)} páginas extraídas.")
    print(f"Arquivo salvo em: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()