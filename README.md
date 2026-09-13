# Palette — Color Season Analyzer

Scan a selfie, get your color season. See [CLAUDE.md](./CLAUDE.md) for product scope and conventions.

## Layout

```
backend/    FastAPI service (Python 3.12)
frontend/   Next.js app (App Router, TypeScript, Tailwind)
```

## Running locally with Docker

Requires Docker with Compose v2+.

```bash
docker compose up --build
```

| Service      | URL                            |
| ------------ | ------------------------------ |
| Frontend     | http://localhost:3000          |
| API          | http://localhost:8000          |
| API docs     | http://localhost:8000/docs     |
| Health check | http://localhost:8000/api/health |
| Postgres     | `localhost:5432` (`palette` / `palette`) |

Both services hot-reload from the mounted source. The home page shows the
backend's health status, so a green "Connected" line means the whole chain is
wired up.

Ports and database credentials can be overridden — copy `.env.example` to
`.env` and edit. Tear down with `docker compose down` (add `-v` to also drop
the Postgres volume).

## Running natively

**Backend**

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000` for the API; override with
`NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

## Tests

```bash
docker compose exec backend pytest     # or, natively: cd backend && pytest
```

## Environment variables

Each `.env.example` documents what that side reads — root for Compose,
`backend/.env.example`, and `frontend/.env.example`. Real `.env` files are
gitignored and must never be committed.
