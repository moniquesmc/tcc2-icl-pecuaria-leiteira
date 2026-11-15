"""
Revisão da tipologia das perguntas (Booleana x Descritiva).

A classificação automática feita pelo Llama 3.2 3B (script 3) concordou com a revisão
em apenas 57% das perguntas (marcou como Booleanas muitas perguntas do tipo "O que é...?"
ou "Como fazer...?"). Por isso a tipologia foi revista com uma regra explícita mais a
revisão manual dos casos que a regra não resolve.

Critério:
- Booleana: pode ser respondida com "sim" ou "não" (ex.: "É possível...?", "Existe...?",
  "Pode-se...?", "Deve-se...?").
- Descritiva: pede explicação, procedimento, quantidade, momento ou escolha entre
  alternativas (ex.: "O que é...?", "Como...?", "Qual...?", "Quantos...?",
  "... A ou B?").

Regra: perguntas que começam com palavra interrogativa (o que, qual, quais, como, por que,
quando, quanto, onde, quem, em que, para que...) são Descritivas; as demais são Booleanas.
Exceções revisadas manualmente estão listadas abaixo.

Uso:
    python scripts/3b_revisar_tipologia.py
"""
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

PASTA = Path(__file__).resolve().parent.parent / "dados" / "processados"
ARQUIVOS = ["perguntas_estruturadas_final.json", "perguntas_estruturadas_parcial.json"]

INTERROGATIVAS = (r"^(o que|qual|quais|como|por que|quando|quanto|quantos|quantas|onde|quem|"
                  r"em que|para que|de que|que |por quanto|ate quando)")

# Revisão manual: perguntas sem palavra interrogativa no início, mas que pedem explicação,
# quantidade, momento ou escolha entre alternativas ("... A ou B?").
DESCRITIVAS_MANUAIS = {19, 21, 38, 50, 62, 63, 71, 75, 76, 93, 97, 104, 106, 114, 134, 146, 149,
                       163, 165, 204, 226, 239, 242, 245, 251, 270, 281, 282, 291, 319, 326, 352,
                       363, 367, 371, 373, 384, 389, 477, 481}
# Revisão manual: começam com "Quando..." condicional, mas a pergunta é de sim/não.
BOOLEANAS_MANUAIS = {72, 150, 433}


def sem_acento(t):
    return "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn").lower()


def tipologia(item):
    if item["id"] in DESCRITIVAS_MANUAIS:
        return "Descritiva"
    if item["id"] in BOOLEANAS_MANUAIS:
        return "Booleana"
    texto = sem_acento(re.sub(r"\s+", " ", item["pergunta"]).strip())
    return "Descritiva" if re.match(INTERROGATIVAS, texto) else "Booleana"


def main():
    for nome in ARQUIVOS:
        caminho = PASTA / nome
        if not caminho.exists():
            continue
        dados = json.load(open(caminho, encoding="utf-8"))
        antes = Counter(d.get("tipologia") for d in dados)
        mudou = 0
        for d in dados:
            nova = tipologia(d)
            if d.get("tipologia") and d.get("tipologia") != nova:
                mudou += 1
            d["tipologia_llm"] = d.get("tipologia_llm") or d.get("tipologia")
            d["tipologia"] = nova
        json.dump(dados, open(caminho, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"{nome}: antes {dict(antes)} -> agora {dict(Counter(d['tipologia'] for d in dados))} "
              f"({mudou} alteradas)")


if __name__ == "__main__":
    main()
