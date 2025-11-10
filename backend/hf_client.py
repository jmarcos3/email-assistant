
import os
import time
from typing import Literal, Optional
import httpx
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

Category = Literal["Produtivo", "Improdutivo"]

HF_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_CLASSIFY_MODEL = os.getenv("HF_CLASSIFY_MODEL", "google/flan-t5-base").strip()
HF_REPLY_MODEL = os.getenv("HF_REPLY_MODEL", "google/flan-t5-base").strip()
HF_TIMEOUT = float(os.getenv("HF_TIMEOUT_SECONDS", "60"))

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def _post_with_fallbacks(task: str, model: str, payload: dict, retries: int = 3) -> Optional[dict]:
    if not HF_TOKEN:
        print("[HF] token ausente")
        return None

    routes = [
        f"https://api-inference.huggingface.co/models/{model}?wait_for_model=true",
        f"https://api-inference.huggingface.co/pipeline/{task}",  
    ]
    last_err = None

    for route in routes:
        for attempt in range(1, retries + 1):
            try:
                body = payload
                if "/pipeline/" in route:
                    body = {"model": model}
                    body.update(payload)

                with httpx.Client(timeout=HF_TIMEOUT) as client:
                    res = client.post(route, headers=HEADERS, json=body)

                    if res.status_code == 503:
                        print(f"[HF] 503 (warm-up) em {route} (tentativa {attempt}/{retries})")
                        time.sleep(min(5 * attempt, 12))
                        continue

                    if res.status_code == 410:
                        print(f"[HF] 410 Gone em {route} — tentando rota de pipeline…")
                        break

                    res.raise_for_status()
                    return res.json()

            except Exception as e:
                last_err = e
                print(f"[HF] erro {route} tentativa {attempt}/{retries}: {e}")
                time.sleep(min(3 * attempt, 8))

    print(f"[HF] falhou após {retries} tentativas. Último erro: {last_err}")
    return None


def _extract_generated_text(data: dict | list) -> Optional[str]:
    if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
        return str(data[0]["generated_text"]).strip()
    if isinstance(data, dict) and "generated_text" in data:
        return str(data["generated_text"]).strip()
    return None


def classify_with_hf(text: str) -> Optional[Category]:
    prompt = (
        "Classifique o e-mail abaixo como 'Produtivo' ou 'Improdutivo'. "
        "Responda APENAS com UMA palavra exatamente igual a uma das opções.\n\n"
        f"E-mail:\n{text}\n\nResposta:"
    )
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 3}}
    data = _post_with_fallbacks("text2text-generation", HF_CLASSIFY_MODEL, payload)
    if not data:
        return None

    out = _extract_generated_text(data)
    if not out:
        return None

    out_l = out.strip().lower()
    if "produtivo" in out_l:
        return "Produtivo"
    if "improdutivo" in out_l:
        return "Improdutivo"
    return None


def generate_reply_with_hf(category: Category, text: str) -> Optional[str]:
    if category == "Produtivo":
        prompt = (
            "Gere uma resposta curta e profissional em PT-BR (máx 3 frases) para este e-mail produtivo: "
            "confirme recebimento, indique a próxima ação e um prazo curto.\n\nE-mail:\n" + text
        )
    else:
        prompt = (
            "Gere uma resposta curta e educada em PT-BR (máx 2 frases) para este e-mail improdutivo "
            "(sem demanda). Agradeça e ofereça canal de suporte se necessário.\n\nE-mail:\n" + text
        )

    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 80}}
    data = _post_with_fallbacks("text2text-generation", HF_REPLY_MODEL, payload)
    if not data:
        return None

    out = _extract_generated_text(data)
    return out if out else None
