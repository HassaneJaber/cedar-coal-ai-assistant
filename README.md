# Cedar & Coal AI Assistant

A multilingual chat assistant for a fictional Lebanese restaurant in Beirut. Built as a portfolio demo of restaurant FAQs, structured booking collection and confirmation-gated storage.

**Live demo:** [Coming soon — Render deployment](#live-demo)

## What the demo does

- Answers menu, opening-hours and delivery questions in English, Lebanese Arabic, Arabizi and French, using a restaurant reference file.
- Collects booking details one field at a time, retains corrections, shows a localized review and saves only after confirmation.
- Adds localized allergy and cross-contact notices. It cannot guarantee that a dish is safe for an allergy.
- Routes groups of 11 or more toward a callback request rather than accepting an online booking.
- Records complaint and callback requests. It does not notify real staff or check real table availability.

This is a fictional demo: bookings do not reserve a real table. Use made-up contact details when trying it.

## Live demo

Deployment URL will be added here after the Render service is live.

## Screenshots

Screenshots will be added after deployment: the multilingual welcome screen, a booking review, and an Arabizi booking conversation.

## Tech stack

Python 3.11, Chainlit, the OpenAI Python SDK with Groq's compatible API, python-dotenv, filelock, local CSV storage, and Docker. An optional Google Sheets adapter is included; the initial Render deployment uses CSV.

The main model is openai/gpt-oss-120b with medium reasoning effort and a 600-token completion cap. A configured qwen/qwen3.8-27b fallback handles daily-quota errors; per-minute limits retry with backoff. Model availability and quotas depend on the provider.

## Architecture

1. **Chat UI — app.py:** creates a session-specific assistant, welcome text and confirmation buttons.
2. **Booking orchestration — assistant.py and booking_state.py:** keeps a validated Python draft independently of the 20-message history window. Recognized booking steps use fixed localized questions without a model call. Off-script requests use the model and tools. Dates/times follow Beirut time, opening hours and kitchen last order.
3. **Safety and language — reply_policy.py and booking_guard.py:** apply allergy/dietary notices, large-group handling, sticky language and checks for repeated fields, Arabic script in Arabizi, and selected non-Lebanese words. These guards reduce specific errors; they do not guarantee all model replies are correct.
4. **Knowledge — knowledge/:** restaurant.md supplies facts. Arabic and Arabizi example files guide style only.
5. **Storage — storage.py:** confirmation gates writes to local CSV, using file locks and spreadsheet-formula escaping. In-memory drafts are lost when the session or app restarts.
6. **Verification — tests/, eval.py and conversation_eval.py:** offline fake-model tests exercise state and guards. Optional single-question evaluation uses an LLM judge; conversation evaluation checks corrections and saved records with deterministic assertions. LLM judging is fallible. Generated transcripts and run ledgers are excluded from the public repository.

## Run locally

Python 3.11+ and a Groq API key are required for model-backed replies. Offline tests do not need a key.

Windows PowerShell:

    git clone https://github.com/HassaneJaber/cedar-coal-ai-assistant.git
    cd cedar-coal-ai-assistant
    py -3.11 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    Copy-Item .env.example .env

Edit .env privately and set GROQ_API_KEY. Do not paste credentials into chat or commit them.
Leave both Google settings empty to use local CSV.

    .\.venv\Scripts\python.exe -m chainlit run app.py --host 127.0.0.1 --port 8000 --headless

Open http://127.0.0.1:8000. On macOS/Linux, use python3 -m venv .venv and .venv/bin/python for the equivalent commands.

## Offline checks and optional evaluation

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

Optional live evaluations use API quota and are separate from offline tests. Run only when intended:

    .\.venv\Scripts\python.exe eval.py --ids 3,4
    .\.venv\Scripts\python.exe eval.py --conversations

The full single-question evaluation (eval.py without flags) and conversation evaluation each have their own once-per-Beirut-day ledger. A fresh clone does not carry those local ledgers; coordinate runs yourself. The latest wording/deployment changes have not been evaluated with live models.

## Docker and Render

Build and run locally when Docker is available:

    docker build -t cedar-coal-demo .
    docker run --rm -p 8000:8000 --env-file .env cedar-coal-demo

The container listens on 0.0.0.0 and uses PORT, defaulting to 8000. The image excludes environment files, local CSVs, caches and evaluation artifacts.

render.yaml defines one free Docker web service named cedar-coal-demo. In Render, choose New > Blueprint, connect this repository, and enter GROQ_API_KEY when prompted. The Blueprint sets the two model names and reasoning effort; leave Google credentials unset for CSV storage.

[Render Blueprint setup](https://render.com/docs/infrastructure-as-code) · [Free-service limitations](https://render.com/docs/free)

Free Render services can sleep when idle, and local CSV files are lost on redeploy, restart or spin-down. This deployment is for demonstration, not durable customer records. Google Sheets integration is a later deployment step.

## Limitations

No real availability checks, payment flow, SMS or staff notifications. Language detection and deterministic guards cover defined patterns, not every paraphrase. Model responses still require review, especially allergy information. Provider quota exhaustion and free-service cold starts can affect response time.
