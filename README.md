# 🌾 Dhan Saathi — AI Rice Advisory Chatbot

End-Term Project · AI Application · Use case #18 (Agri-advisory bot for a specific crop/region)

A Streamlit chatbot that gives rice (paddy) farmers in Punjab, Haryana and Delhi-NCR stage-aware,
weather-linked advice in English, Hindi or Punjabi, using Google Gemini grounded on a curated knowledge base.

## Features
- 💬 Multi-turn chat with memory (last 8 messages) and quick-question buttons
- 📚 Grounded answers: BM25 retrieval over 28 knowledge-base notes; every answer shows Sources + Confidence
- 🌦️ Weather-linked advisory: live 5-day forecast (Open-Meteo, no key) with sample-data fallback + 8 fixed agronomy rules
- 🐛 Guided pest & disease check (symptom checklist + optional photo, Gemini vision)
- 🗣️ English / हिंदी / ਪੰਜਾਬੀ
- 🛡️ Guardrails: scope limit, prompt-injection filter, no pesticide doses, poisoning emergency message, AI disclosure
- 🧑‍🌾 Escalation to human expert (Kisan Call Centre 1800-180-1551 / KVK) with demo ticket ID
- 🔁 Failure handling: retries, automatic model fallback, offline knowledge-base mode if Gemini is down

## Project files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI, prompt design, Gemini calls, fallbacks |
| `engine.py` | Retrieval, crop-stage logic, weather rules, input guardrails (no AI) |
| `data/knowledge_base.json` | 28 sample rice advisory notes (KB01–KB28) |
| `data/sample_weather.json` | Sample 5-day forecasts for 6 districts (used if live weather fails) |
| `requirements.txt` | Python packages |
| `.streamlit/secrets.toml.example` | Template for your API key |

---

## Step 1 — Get a free Gemini API key (2 min)
1. Go to **https://aistudio.google.com** and sign in with a Google account.
2. Click **Get API key → Create API key**. Copy it. Keep it private.
3. In AI Studio, note which Flash models your key can use (the app tries
   `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.5-flash-lite`, `gemini-2.5-flash` in order).

## Step 2 — (Optional) Run on your laptop
```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key inside
streamlit run app.py
```
Opens at http://localhost:8501.

## Step 3 — Put the code on GitHub
1. Create a free account at **github.com** → **New repository** → name it `dhan-saathi` → Public → Create.
2. Click **uploading an existing file** and drag in: `app.py`, `engine.py`, `requirements.txt`,
   `README.md`, the `data` folder, and the `.streamlit` folder (**only** `config.toml` and `secrets.toml.example` —
   never a file containing your real key). Commit.

## Step 4 — Deploy on Streamlit Community Cloud (free shareable link)
1. Go to **https://share.streamlit.io** → sign in with GitHub.
2. **Create app → Deploy a public app from GitHub** → choose your repo, branch `main`, main file `app.py`.
3. Click **Advanced settings → Secrets** and paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
4. Choose a custom URL (e.g. `dhan-saathi-pravneet`) and click **Deploy**. In ~2 minutes you get a link like
   `https://dhan-saathi-pravneet.streamlit.app` — that is the link to submit.
5. Open the link and check the sidebar shows **🟢 Gemini connected**.

> Free apps sleep after a period of no use. Open the link once before your presentation so it wakes up.

### Troubleshooting
| Symptom | Fix |
|---|---|
| Sidebar says 🟠 Offline mode | Secret not set or misspelt — must be exactly `GEMINI_API_KEY` |
| "AI model unavailable" | Add `GEMINI_MODEL = "<a model listed in your AI Studio>"` to secrets |
| "AI quota reached" | Free-tier per-minute limit; wait a minute |
| Weather says "sample data" | Open-Meteo unreachable; app correctly fell back to bundled sample data |
