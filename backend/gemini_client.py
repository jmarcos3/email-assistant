import os
import time
from typing import Literal, Optional
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
import re

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

Category = Literal["Produtivo", "Improdutivo"]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.3"))

if not GOOGLE_API_KEY:
    print("[Gemini] ⚠️ Nenhuma API key configurada em GOOGLE_API_KEY")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

_model = genai.GenerativeModel(GEMINI_MODEL)

def _generate(prompt: str, max_tokens: int = 256, retries: int = 2) -> Optional[str]:
    if not GOOGLE_API_KEY:
        return None

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            response = _model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": max_tokens,
                    "temperature": GEMINI_TEMPERATURE,
                },
            )
            return (response.text or "").strip()
        except Exception as e:
            last_err = e
            print(f"[Gemini] Erro tentativa {attempt}/{retries}: {e}")
            time.sleep(2 * attempt)
    print(f"[Gemini] Falhou após {retries} tentativas. Último erro: {last_err}")
    return None


def classify_with_gemini(text: str) -> Optional[Category]:
    prompt = f"""
                Classifique o e-mail abaixo como 'Produtivo' ou 'Improdutivo'. 
                Responda APENAS com UMA palavra exatamente igual a uma das opções.

                E-mail:
                {text}

                Resposta:
            """
    out = _generate(prompt, max_tokens=4)
    if not out:
        return None

    low = out.lower()
    m = re.search(r"\b(improdutivo|produtivo)\b", low)
    if not m: return None
    return "Improdutivo" if m.group(1) == "improdutivo" else "Produtivo"


def generate_reply_with_gemini(category: Category, text: str) -> Optional[str]:
    if category == "Produtivo":
        prompt = f"""
                Gere uma resposta curta e profissional em PT-BR (máx 3 frases) para este e-mail produtivo.
                Confirme recebimento, indique a próxima ação e um prazo curto.

                E-mail:
                {text}
            """
    else:
        prompt = f"""
                    Gere uma resposta curta e educada em PT-BR (máx 2 frases) para este e-mail improdutivo (sem demanda).
                    Agradeça e ofereça canal de suporte se necessário.

                    E-mail:
                    {text}
                """
    out = _generate(prompt, max_tokens=100)
    return out if out else None
