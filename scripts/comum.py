"""
Funções compartilhadas por todos os experimentos de In-Context Learning (ICL).

Centraliza: categorias, chamadas aos 4 LLMs, leitura das respostas, montagem dos
prompts, seleção de demonstrações, embeddings e salvamento/retomada dos resultados.
Assim, todas as técnicas usam exatamente as mesmas condições experimentais e só
diferem naquilo que caracteriza cada técnica.
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# ======================= CAMINHOS =======================
DIRETORIO_SCRIPTS = Path(__file__).resolve().parent
PASTA_DADOS = DIRETORIO_SCRIPTS.parent / "dados" / "processados"
ARQUIVO_FOLDS = PASTA_DADOS / "folds_cv.json"
PASTA_RESULTADOS = PASTA_DADOS / "resultados"
PASTA_CACHE = PASTA_DADOS / "cache_respostas"   # cada resposta do LLM fica salva aqui

# ======================= CONFIGURAÇÃO DO EXPERIMENTO =======================
CATEGORIAS = [
    "Cria e Recria de Bezerras e Novilhas", "Alimentação e Manejo de Vacas e Touros",
    "Recursos Forrageiros", "Reprodução", "Melhoramento Genético Animal",
    "Saúde Animal", "Mastite e Qualidade do Leite", "Produção Orgânica de Leite",
    "Instalações, Ambiência e Manejo de Dejetos", "Bem-estar Animal",
    "Gerenciamento da Atividade Leiteira",
]

MODELOS_PARA_TESTAR = ["ollama", "qwen", "llama-70b", "llama-nvidia", "gemini"]

MODELO_OLLAMA = "llama3.2:3b"
MODELO_GROQ = "qwen/qwen3.8-27b"
MODELO_QWEN_OPENROUTER = "qwen/qwen3.8-27b"   # mesmo modelo, via OpenRouter (upgrade do Groq indisponível)
MODELO_NVIDIA = "meta/llama-3.2-90b-vision-instruct"
MODELO_LLAMA70_OPENROUTER = "meta-llama/llama-3.3-70b-instruct"   # alternativa ao 90B (NVIDIA lenta)
MODELO_GEMINI = "gemini-3.1-flash-lite"
MODELO_GEMINI_OPENROUTER = "google/gemini-3.1-flash-lite"
# Com a chave do OpenRouter no .env, o Gemini é chamado por ele (pago por token, sem limite
# diário do plano gratuito do Google). Sem a chave, usa a API gratuita do Google.
GEMINI_VIA_OPENROUTER = bool(os.getenv("OPENROUTER_API_KEY"))

# Modelo de embeddings MULTILÍNGUE (as perguntas estão em português).
MODELO_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2"

SEMENTE = 42                 # semente base (reprodutibilidade)
K_KATE = 5                   # nº de vizinhos no KATE / Ordenação
N_VALIDACAO = 20             # perguntas da validação usadas por APE e GlobalE/LocalE
N_CANDIDATOS_APE = 5         # nº de instruções candidatas geradas no APE
N_PERMUTACOES = 6            # nº de ordens testadas no GlobalE/LocalE
ENTRADAS_NULAS = ["N/A", "", "[MASK]"]   # entradas sem conteúdo da Calibração (Zhao et al., 2021)
PAUSA_ENTRE_CHAMADAS = 1.5   # segundos
# Intervalo maior para a API gratuita do Gemini (limite por minuto)
PAUSA_GEMINI_GRATUITO = 4.5

CHAVES = {
    "groq": os.getenv("GROQ_API_KEY"),
    "nvidia": os.getenv("NVIDIA_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "openrouter": os.getenv("OPENROUTER_API_KEY"),
}

INSTRUCAO_PADRAO = (
    "Classifique a dúvida do produtor rural em EXATAMENTE UMA das 11 categorias técnicas abaixo."
)


# ======================= CHAMADAS AOS LLMs =======================
def verificar_resposta(resp):
    """Como raise_for_status, mas inclui a mensagem da API (ex.: qual limite foi atingido)."""
    if resp.status_code >= 400:
        erro = requests.HTTPError(f"{resp.status_code} {resp.reason}: {resp.text[:400]}", response=resp)
        raise erro


def chamar_llm(prompt, modelo):
    if modelo == "ollama":
        url = "http://localhost:11434/api/generate"
        # num_predict limita o tamanho da resposta: o modelo pequeno às vezes entra em
        # repetição e não para de gerar (estouro de tempo). 512 tokens é bem mais do que
        # qualquer resposta válida precisa (a maior, do SG-ICL, tem ~300 tokens).
        payload = {"model": MODELO_OLLAMA, "prompt": prompt, "stream": False,
                   "options": {"temperature": 0.0, "num_predict": 512}}
        resp = requests.post(url, json=payload, timeout=180)
        verificar_resposta(resp)
        return resp.json().get("response", "").strip()

    if modelo == "qwen-groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['groq']}"}
        payload = {"model": MODELO_GROQ, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0}
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        verificar_resposta(resp)
        return (resp.json()["choices"][0]["message"].get("content") or "").strip()

    if modelo == "qwen":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['openrouter']}"}
        payload = {"model": MODELO_QWEN_OPENROUTER,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0,
                   "reasoning": {"enabled": False}}     # sem "thinking": resposta direta, como nos demais
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        verificar_resposta(resp)
        dados = resp.json()
        if "choices" not in dados:
            raise RuntimeError(f"Resposta inesperada do OpenRouter: {str(dados)[:300]}")
        return (dados["choices"][0]["message"].get("content") or "").strip()

    if modelo == "llama-70b":
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['openrouter']}"}
        payload = {"model": MODELO_LLAMA70_OPENROUTER,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0}
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        verificar_resposta(resp)
        dados = resp.json()
        if "choices" not in dados:
            raise RuntimeError(f"Resposta inesperada do OpenRouter: {str(dados)[:300]}")
        return (dados["choices"][0]["message"].get("content") or "").strip()

    if modelo == "llama-nvidia":
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['nvidia']}"}
        payload = {"model": MODELO_NVIDIA, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0}
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        verificar_resposta(resp)
        return (resp.json()["choices"][0]["message"].get("content") or "").strip()

    if modelo == "gemini" and GEMINI_VIA_OPENROUTER:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {CHAVES['openrouter']}"}
        payload = {"model": MODELO_GEMINI_OPENROUTER,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0}
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        verificar_resposta(resp)
        dados = resp.json()
        if "choices" not in dados:
            raise RuntimeError(f"Resposta inesperada do OpenRouter: {str(dados)[:300]}")
        return (dados["choices"][0]["message"].get("content") or "").strip()

    if modelo == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO_GEMINI}:generateContent"
        headers = {"Content-Type": "application/json", "x-goog-api-key": CHAVES["gemini"]}
        payload = {"contents": [{"parts": [{"text": prompt}]}],
                   "generationConfig": {"temperature": 0.0}}
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        verificar_resposta(resp)
        candidato = resp.json().get("candidates", [{}])[0]
        partes = candidato.get("content", {}).get("parts", [])
        return " ".join(p.get("text", "") for p in partes if not p.get("thought")).strip()

    raise ValueError(f"Modelo desconhecido: {modelo}")


class FalhaDeAPI(RuntimeError):
    """Lançada quando a API não responde (limite diário, internet, servidor). Interrompe o
    script; ao rodar de novo, as respostas já obtidas vêm do cache e nada é perdido."""


# ---- Cache: cada resposta é gravada no disco assim que chega. Ao rodar de novo, um prompt
# ---- já respondido NÃO é reenviado (economiza cota e retoma exatamente de onde parou).
_cache = {}


def _cache_do_modelo(modelo):
    if modelo not in _cache:
        _cache[modelo] = {}
        arquivo = PASTA_CACHE / f"{modelo}.jsonl"
        if arquivo.exists():
            with open(arquivo, "r", encoding="utf-8") as f:
                for linha in f:
                    try:
                        item = json.loads(linha)
                        _cache[modelo][item["h"]] = item["r"]
                    except (json.JSONDecodeError, KeyError):
                        continue   # linha incompleta (ex.: PC desligou no meio da gravação)
    return _cache[modelo]


def _guardar_no_cache(modelo, chave, resposta):
    PASTA_CACHE.mkdir(parents=True, exist_ok=True)
    with open(PASTA_CACHE / f"{modelo}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"h": chave, "r": resposta}, ensure_ascii=False) + "\n")
    _cache[modelo][chave] = resposta


def _limite_diario(msg):
    msg = msg.lower()
    return any(t in msg for t in ("per day", "perday", "(rpd)", "(tpd)", "daily"))


ESPERA_MAX_LIMITE_DIARIO = 900   # segundos: acima disso, para e pede para rodar mais tarde


def _espera_no_texto(msg):
    """Lê 'try again in 20.3s' / 'in 7m30.5s' / 'in 1h2m' da mensagem de erro."""
    m = re.search(r"try again in\s*(?:(\d+)h)?\s*(?:(\d+)m(?!s))?\s*(?:([\d.]+)s)?", msg, re.I)
    if not m or not any(m.groups()):
        return None
    h, mi, se = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(se or 0)


def _espera_sugerida(erro):
    resp = getattr(erro, "response", None)
    if resp is not None:
        try:
            return min(600, max(5, float(resp.headers.get("retry-after", "")) + 2))
        except ValueError:
            pass
    return None


def chamar_com_retry(prompt, modelo, max_tentativas=8):
    cache = _cache_do_modelo(modelo)
    chave = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if chave in cache:
        return cache[chave]

    tentativa = 0
    while tentativa < max_tentativas:
        tentativa += 1
        try:
            resposta = chamar_llm(prompt, modelo)
            _guardar_no_cache(modelo, chave, resposta)
            gratuito = modelo == "gemini" and not GEMINI_VIA_OPENROUTER
            time.sleep(PAUSA_GEMINI_GRATUITO if gratuito else PAUSA_ENTRE_CHAMADAS)
            return resposta
        except Exception as e:
            msg = str(e)
            print(f"\n  [!] Erro na API ({modelo}), tentativa {tentativa}/{max_tentativas}: {msg[:300]}")
            if "429" in msg and _limite_diario(msg):
                espera_dia = _espera_no_texto(msg)
                if espera_dia is not None and espera_dia <= ESPERA_MAX_LIMITE_DIARIO:
                    # cota diária em janela móvel (ex.: Groq): libera aos poucos -> espera e segue
                    print(f"  Cota diária quase esgotada; a API libera de novo em {espera_dia:.0f} s. Aguardando...")
                    time.sleep(espera_dia + 2)
                    tentativa -= 1     # esperar a cota não conta como falha
                    continue
                raise FalhaDeAPI(f"Limite DIÁRIO da API de '{modelo}' atingido. Rode de novo "
                                 "amanhã (ou quando a cota renovar): nada do que já foi feito se perde.")
            if tentativa == max_tentativas:
                break
            espera = _espera_sugerida(e) or (65 if "429" in msg else 20)
            print(f"  Aguardando {espera:.0f} s para tentar novamente...")
            time.sleep(espera)
    raise FalhaDeAPI(
        f"A API do modelo '{modelo}' falhou {max_tentativas} vezes seguidas "
        "(limite, internet ou servidor fora do ar). Rode o script de novo mais tarde: "
        "as respostas já obtidas estão salvas e não serão pedidas de novo."
    )


# ======================= LEITURA DAS RESPOSTAS =======================
def sem_acento(texto):
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def remover_raciocinio(texto):
    """Remove o raciocínio interno (<think>...</think>) de modelos como o Qwen 3."""
    texto = texto or ""
    texto = re.sub(r"<think>.*?</think>", " ", texto, flags=re.S | re.I)
    texto = re.sub(r"<think>.*", " ", texto, flags=re.S | re.I)
    return texto.strip()


def normalizar(texto):
    texto = remover_raciocinio(texto)
    texto = texto.replace('"', "").replace("'", "").replace("*", "").replace("#", "")
    return sem_acento(texto).lower()


CATEGORIAS_NORMALIZADAS = [(c, sem_acento(c).lower()) for c in CATEGORIAS]


def primeira_categoria(texto_normalizado):
    """Categoria que aparece PRIMEIRO no texto (posição), ou None."""
    melhor, melhor_pos = None, None
    for cat, cat_norm in CATEGORIAS_NORMALIZADAS:
        pos = texto_normalizado.find(cat_norm)
        if pos != -1 and (melhor_pos is None or pos < melhor_pos):
            melhor, melhor_pos = cat, pos
    return melhor


def extrair_classe_prevista(resposta_llm):
    """Extrai a categoria escolhida. Se houver 'CATEGORIA FINAL:', usa o texto após a última."""
    if not resposta_llm:
        return "Erro de Formatação"
    texto = normalizar(resposta_llm)
    marcas = list(re.finditer(r"categoria final\s*:", texto))
    if marcas:
        texto = texto[marcas[-1].end():]
    return primeira_categoria(texto) or "Erro de Formatação"


def extrair_distribuicao(resposta_llm, suavizacao=0.0):
    """
    Lê linhas no formato '<categoria>: <número>%' e devolve um vetor de probabilidades
    (np.array, na ordem de CATEGORIAS, somando 1). Categorias ausentes recebem 0.
    Retorna None se nenhuma probabilidade puder ser lida.
    """
    texto = normalizar(resposta_llm)
    valores = {}
    for linha in texto.splitlines():
        cat = primeira_categoria(linha)
        if cat is None or cat in valores:
            continue
        resto = linha[linha.find(sem_acento(cat).lower()) + len(cat):]
        numero = re.search(r"(\d+(?:[.,]\d+)?)\s*%", resto) or re.search(r"(\d+(?:[.,]\d+)?)", resto)
        if numero:
            valores[cat] = float(numero.group(1).replace(",", "."))
    vetor = np.array([valores.get(c, 0.0) for c in CATEGORIAS], dtype=float)
    if vetor.sum() <= 0:
        return None
    vetor = vetor / vetor.sum()
    if suavizacao > 0:
        vetor = (vetor + suavizacao) / (1 + suavizacao * len(vetor))
    return vetor


def entropia(probabilidades):
    p = np.asarray(probabilidades, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ======================= MONTAGEM DOS PROMPTS =======================
def lista_categorias():
    # separador ';' porque uma das categorias contém vírgula
    return "; ".join(CATEGORIAS)


def formatar_demonstracoes(demonstracoes):
    return "".join(f"Dúvida: {d['pergunta']}\nCategoria: {d['categoria_tematica']}\n\n"
                   for d in demonstracoes)


def montar_prompt(pergunta, demonstracoes=(), instrucao=INSTRUCAO_PADRAO):
    """
    Prompt padrão de classificação usado por TODAS as técnicas.
    A dúvida a classificar fica sempre por último, logo após as demonstrações.
    """
    prompt = (
        f"{instrucao}\n"
        f"Categorias válidas: {lista_categorias()}.\n"
        "Responda APENAS com o nome de uma das categorias, sem justificativas.\n\n"
    )
    if demonstracoes:
        prompt += "Exemplos:\n" + formatar_demonstracoes(demonstracoes)
    prompt += f"Dúvida: {pergunta}\nCategoria:"
    return prompt


def montar_prompt_probabilidades(pergunta, demonstracoes=(), instrucao=INSTRUCAO_PADRAO):
    """Mesmo prompt, mas pede a probabilidade de CADA categoria (usado em LocalE/GlobalE e Calibração)."""
    prompt = (
        f"{instrucao}\n"
        f"Categorias válidas: {lista_categorias()}.\n\n"
    )
    if demonstracoes:
        prompt += "Exemplos:\n" + formatar_demonstracoes(demonstracoes)
    prompt += (
        f"Dúvida: {pergunta}\n\n"
        "Para a dúvida acima, estime a probabilidade (de 0 a 100%) de ela pertencer a CADA uma "
        "das 11 categorias. As probabilidades devem somar 100%.\n"
        "Responda exatamente em 11 linhas, sem explicações, no formato:\n"
        "<nome da categoria>: <probabilidade>%"
    )
    return prompt


# ======================= SELEÇÃO DE DEMONSTRAÇÕES =======================
def selecionar_fewshot(treino, fold):
    """
    1 demonstração aleatória por categoria (11 no total), em ordem aleatória.
    Semente fixa por fold: Few-shot, GlobalE, LocalE e Calibração usam
    EXATAMENTE as mesmas demonstrações, o que permite comparar só o efeito de cada técnica.
    """
    rng = random.Random(SEMENTE + fold)
    por_classe = {c: [] for c in CATEGORIAS}
    for item in treino:
        por_classe[item["categoria_tematica"]].append(item)
    demonstracoes = [rng.choice(por_classe[c]) for c in CATEGORIAS if por_classe[c]]
    rng.shuffle(demonstracoes)
    return demonstracoes


def amostra_validacao(validacao, fold, n=N_VALIDACAO):
    """Subconjunto fixo da validação (estratificado por sorteio simples com semente)."""
    rng = random.Random(SEMENTE + 1000 + fold)
    return rng.sample(validacao, min(n, len(validacao)))


_modelo_embedding = None


def carregar_modelo_embedding():
    global _modelo_embedding
    if _modelo_embedding is None:
        from sentence_transformers import SentenceTransformer
        print(f"Carregando modelo de embeddings ({MODELO_EMBEDDING})...")
        _modelo_embedding = SentenceTransformer(MODELO_EMBEDDING)
    return _modelo_embedding


class IndiceSemantico:
    """Calcula os embeddings do treino UMA vez por fold e busca os vizinhos mais similares."""

    def __init__(self, treino):
        self.treino = treino
        self.modelo = carregar_modelo_embedding()
        self.vetores = self.modelo.encode([t["pergunta"] for t in treino], normalize_embeddings=True)

    def vizinhos(self, pergunta, k):
        """k exemplos mais similares, do MAIS para o MENOS similar."""
        v = self.modelo.encode([pergunta], normalize_embeddings=True)[0]
        similaridades = self.vetores @ v
        indices = np.argsort(-similaridades)[:k]
        return [self.treino[i] for i in indices], [float(similaridades[i]) for i in indices]


# ======================= RESULTADOS (salvar / retomar) =======================
def arquivo_resultados(grupo, modelo):
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    return PASTA_RESULTADOS / f"{grupo}_{modelo}.json"


def carregar_resultados(arquivo):
    if arquivo.exists():
        with open(arquivo, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def salvar_resultados(arquivo, folds_concluidos):
    temporario = arquivo.with_suffix(".tmp")
    with open(temporario, "w", encoding="utf-8") as f:
        json.dump(folds_concluidos, f, ensure_ascii=False, indent=2)
    os.replace(temporario, arquivo)   # gravação atômica: não corrompe se o PC desligar


def carregar_folds():
    with open(ARQUIVO_FOLDS, "r", encoding="utf-8") as f:
        return json.load(f)


def modelos_da_linha_de_comando(descricao):
    parser = argparse.ArgumentParser(description=descricao)
    parser.add_argument("--modelo", choices=MODELOS_PARA_TESTAR, action="append",
                        help="Rodar só este modelo (pode repetir). Padrão: os 4 modelos.")
    args = parser.parse_args()
    return args.modelo or MODELOS_PARA_TESTAR


def executar_experimento(grupo, descricao, processar_fold):
    """
    Laço padrão: para cada modelo e cada fold ainda não concluído, chama
    processar_fold(fold, modelo) -> dict e salva o resultado ao final do fold.
    """
    modelos = modelos_da_linha_de_comando(descricao)
    folds = carregar_folds()
    for modelo in modelos:
        arquivo = arquivo_resultados(grupo, modelo)
        concluidos = carregar_resultados(arquivo)
        if len(concluidos) >= len(folds):
            print(f"[OK] {modelo.upper()} já concluiu '{grupo}'. Pulando...")
            continue
        print(f"\n{'=' * 60}\n{descricao} | MODELO: {modelo.upper()}\n{'=' * 60}")
        for fold in folds[len(concluidos):]:
            print(f"\n--- {modelo.upper()} | Fold {fold['fold']} ---")
            try:
                resultado = processar_fold(fold, modelo)
            except FalhaDeAPI as e:
                print(f"\n[PARADO] {e}")
                print(f"Folds salvos de {modelo.upper()}: {len(concluidos)}/{len(folds)}.")
                break
            concluidos.append(resultado)
            salvar_resultados(arquivo, concluidos)
            print(f">>> Fold {fold['fold']} de {modelo.upper()} salvo em {arquivo.name}")
    print("\nFim da execução.")


def progresso(modelo, fold, i, total, rotulo=""):
    print(f"[{modelo.upper()} - Fold {fold}] {rotulo}{i}/{total}", end="\r", flush=True)
