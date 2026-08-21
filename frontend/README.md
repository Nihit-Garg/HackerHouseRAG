# Lumina AI — Frontend

React + Vite client for the HackerHouseRAG voice/text RAG pipeline.

## Pages

- **About** — what Lumina is, what it will and won't do
- **Ask** — text or voice question box, live answer with sources
- **History** — every past question and answer, stored locally in the browser
- **Status** — backend health, knowledge base size, last query timing

## Running it

```bash
npm install
cp .env.example .env
npm run dev
```

The backend must be running separately (`uvicorn api:app --reload` from `src/`, see the repo root README). `VITE_API_BASE_URL` in `.env` points the frontend at it — defaults to `http://localhost:8000`.

## Notes

- No accounts, no server-side history — everything in History/Status is read from `localStorage` on this device only, matching what the backend actually persists (nothing).
- Voice input uses the browser's microphone (`MediaRecorder`) or an uploaded audio file, both sent to `/query/audio`.
