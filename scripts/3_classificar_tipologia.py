import json
import os
import requests
import time
from pathlib import Path
from dotenv import load_dotenv

# Carrega as chaves de API do arquivo .env
load_dotenv()

DIRETORIO_ATUAL = Path(__file__).resolve().parent
ARQUIVO_ENTRADA = DIRETORIO_ATUAL.parent / "dados" / "processados" / "perguntas_estruturadas_parcial.json"
ARQUIVO_SAIDA = DIRETORIO_ATUAL.parent / "dados" / "processados" / "perguntas_estruturadas_final.json"

# Configurações das APIs
CHAVES = {
    "groq": os.getenv("GROQ_API_KEY"),
    "nvidia": os.getenv("NVIDIA_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY")
}

def chamar_llm(prompt, modelo="ollama"):
    """
    Função unificada para chamar qualquer um dos 4 modelos.
    """
    # 1. Llama 3.2 3B (Local via Ollama)
    if modelo == "ollama":
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "llama3.2:3b", # ou o nome exato que está no seu Ollama (ex: llama32-3b-tcc)
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0 # Temperatura zero para não inventar palavras
        }
        resposta = requests.post(url, json=payload)
        return resposta.json().get("response", "").strip()

    # 2. Qwen 3.8-27B (Nuvem via Groq)
    elif modelo == "qwen-groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['groq']}"}
        payload = {
            "model": "qwen/qwen3.8-27b", # Ajuste para o ID exato do Qwen no Groq
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        resposta = requests.post(url, headers=headers, json=payload)
        return resposta.json()["choices"][0]["message"]["content"].strip()

    # 3. Llama 3.2 90B (Nuvem via NVIDIA)
    elif modelo == "llama-nvidia":
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['nvidia']}"}
        payload = {
            "model": "meta/llama-3.2-90b-vision-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }
        resposta = requests.post(url, headers=headers, json=payload)
        return resposta.json()["choices"][0]["message"]["content"].strip()

    # 4. Gemini 3.1 Flash-Lite (Nuvem via Google API)
    elif modelo == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent?key={CHAVES['gemini']}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0}
        }
        resposta = requests.post(url, headers=headers, json=payload)
        texto = resposta.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        return texto.strip()

    return "Erro: Modelo não configurado."

def main():
    print("Iniciando a classificação de tipologia das 500 perguntas...")
    print("Certifique-se de que o Ollama está rodando no seu computador!")
    
    with open(ARQUIVO_ENTRADA, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    # Vamos usar o Ollama (Llama 3.2 3B local) para não estourar os limites gratuitos das APIs online
    modelo_escolhido = "ollama" 

    for item in dados:
        if not item["tipologia"]: # Se estiver vazio, classifica
            prompt = (
                f"Leia a seguinte pergunta de um produtor rural sobre pecuária: '{item['pergunta']}'\n\n"
                "Classifique a tipologia dessa pergunta APENAS com uma destas duas palavras:\n"
                "- 'Booleana' (se for uma pergunta de 'Sim' ou 'Não', ou que busca aprovação/confirmação).\n"
                "- 'Descritiva' (se for uma pergunta de 'Como', 'O que', 'Qual', 'Por que', pedindo explicação).\n\n"
                "Sua resposta final deve conter UNICAMENTE a palavra 'Booleana' ou 'Descritiva'."
            )
            
            try:
                resposta_llm = chamar_llm(prompt, modelo=modelo_escolhido)
                
                # Força a limpeza para garantir que o modelo não tagarelou
                if "boolean" in resposta_llm.lower():
                    item["tipologia"] = "Booleana"
                else:
                    item["tipologia"] = "Descritiva"
                    
                print(f"[{item['id']}/500] Classificado como: {item['tipologia']}")
            
            except Exception as e:
                print(f"[{item['id']}/500] Erro ao classificar: {e}")
            
            # Pequena pausa opcional para não superaquecer sua máquina local
            time.sleep(0.1)

    # Salva a base definitiva e finalizada
    with open(ARQUIVO_SAIDA, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    print(f"\nBase estruturada 100% concluída e salva em: {ARQUIVO_SAIDA}")

if __name__ == "__main__":
    main()