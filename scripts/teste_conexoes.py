import os
import requests
from dotenv import load_dotenv

# Força o recarregamento do .env
load_dotenv(override=True)

CHAVES = {
    "groq": os.getenv("GROQ_API_KEY"),
    "nvidia": os.getenv("NVIDIA_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY")
}

MODELO_GEMINI = "gemini-3.1-flash-lite"

def listar_modelos_gemini():
    """Mostra os modelos do Gemini disponíveis para a sua chave."""
    url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
    try:
        r = requests.get(url, headers={"x-goog-api-key": CHAVES["gemini"]}, timeout=60)
        if r.status_code != 200:
            print(f"Não foi possível listar os modelos ({r.status_code}): {r.text[:300]}")
            return
        nomes = [m["name"].replace("models/", "") for m in r.json().get("models", [])
                 if "generateContent" in m.get("supportedGenerationMethods", [])]
        print("Modelos Gemini disponíveis para generateContent:")
        for n in nomes:
            print("  -", n)
    except Exception as e:
        print(f"Erro ao listar modelos: {e}")

def checar_chaves():
    print("--- RAIO-X DAS CHAVES ---")
    for nome, chave in CHAVES.items():
        if not chave:
            print(f"❌ {nome.upper()}: NÃO ENCONTRADA no .env!")
        else:
            print(f"✅ {nome.upper()}: Carregada ({chave[:6]}...{chave[-4:]})")
    print("-------------------------\n")

def chamar_llm(prompt, modelo):
    try:
        # 1. Ollama (Local) - Timeout aumentado para 60s
        if modelo == "ollama":
            url = "http://localhost:11434/api/generate"
            payload = {"model": "llama3.2:3b", "prompt": prompt, "stream": False}
            resposta = requests.post(url, json=payload, timeout=180)
            if resposta.status_code != 200: return f"Erro API: {resposta.text}"
            return resposta.json().get("response", "").strip()

        # 2. Groq (Qwen)
        elif modelo == "qwen-groq":
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CHAVES['groq']}"}
            payload = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": prompt}]}
            resposta = requests.post(url, headers=headers, json=payload, timeout=180)
            if resposta.status_code != 200: return f"Erro API ({resposta.status_code}): {resposta.text}"
            return resposta.json()["choices"][0]["message"]["content"].strip()

        # 3. NVIDIA (Llama Vision)
        elif modelo == "llama-nvidia":
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CHAVES['nvidia']}"}
            payload = {"model": "meta/llama-3.2-90b-vision-instruct", "messages": [{"role": "user", "content": prompt}]}
            resposta = requests.post(url, headers=headers, json=payload, timeout=180)
            if resposta.status_code != 200: return f"Erro API ({resposta.status_code}): {resposta.text}"
            return resposta.json()["choices"][0]["message"]["content"].strip()

        # Qwen3.8 27B via OpenRouter
        elif modelo == "qwen":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {CHAVES['openrouter']}"}
            payload = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.0, "reasoning": {"enabled": False}}
            resposta = requests.post(url, headers=headers, json=payload, timeout=180)
            if resposta.status_code != 200: return f"Erro API ({resposta.status_code}): {resposta.text[:400]}"
            return resposta.json()["choices"][0]["message"].get("content", "").strip()

        # 4. Gemini
        elif modelo == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO_GEMINI}:generateContent"
            headers = {"Content-Type": "application/json", "x-goog-api-key": CHAVES["gemini"]}
            payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.0}}
            resposta = requests.post(url, headers=headers, json=payload, timeout=180)
            if resposta.status_code != 200: return f"Erro API ({resposta.status_code}): {resposta.text}"
            partes = resposta.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
            return " ".join(p.get("text", "") for p in partes if not p.get("thought")).strip()

    except Exception as e:
        return f"ERRO FATAL: {e}"

def main():
    checar_chaves()
    
    print("Testando a conexão com os 4 modelos...\n")
    modelos = ["ollama", "qwen", "llama-nvidia", "gemini"]
    prompt_teste = "Responda apenas com a palavra: OK"

    for mod in modelos:
        print(f"Testando {mod.upper()}...")
        resposta = chamar_llm(prompt_teste, mod)
        print(f"Resposta: {resposta}\n{'-'*40}")
        if mod == "gemini" and "404" in resposta:
            listar_modelos_gemini()

if __name__ == "__main__":
    main() 