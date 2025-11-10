# 📨 AutoU Email Assistant

Aplicação web simples para **classificação de e-mails** e **sugestão de respostas automáticas**, desenvolvida como parte do desafio prático da AutoU.

A aplicação permite o upload de arquivos `.txt` ou `.pdf` (ou a inserção direta de texto) e identifica se o e-mail é **Produtivo** ou **Improdutivo**, exibindo uma resposta sugerida com base na categoria detectada.

---

## 🚀 Tecnologias

- **Backend:** Python + FastAPI  
- **Frontend:** HTML + CSS + JS puro  
- **IA (tentativa):** Hugging Face Inference API  
- **Hospedagem:** (a definir)

---

## ⚙️ Como rodar localmente

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/jmarcos3/autou-email-assistant
cd autou-email-assistant/backend

python3 -m venv .venv
source .venv/bin/activate   # Linux
# ou
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```
### 2️⃣ Configurar variáveis de ambiente
Crie um arquivo `.env` na pasta `backend/`.

Exemplo:
```ini
HF_API_TOKEN=
HF_CLASSIFY_MODEL=google/flan-t5-base
HF_REPLY_MODEL=google/flan-t5-base
HF_TIMEOUT_SECONDS=60
```


### 3️⃣ Iniciar o servidor backend
```bash uvicorn app:app --reload```

O backend rodará em **http://127.0.0.1:8000**

### 4️⃣ Rodar o frontend
Em outro terminal:
```bash 
cd ../frontend
python3 -m http.server 5173
```
Acesse **http://localhost:5173** no navegador.