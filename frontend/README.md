# Adaptive AI NIDS — Frontend Stage 1

React + Vite frontend foundation. This stage implements login/session handling and an honest dashboard-unavailable state because the running backend currently exposes no dashboard summary endpoint.

## Requirements

- Node.js 20+ (Node 22 recommended)
- FastAPI backend at `http://127.0.0.1:8000` for live login/session testing

## Windows CMD setup

From the repository root:

```cmd
cd /d "D:\VS_CODE\AI-Driven IDS\adaptive-nids\frontend"
copy .env.example .env
npm install
npm run test
npm run build
npm run dev
```

Open the URL Vite prints, normally `http://localhost:5173`.

## Authentication notes

The API client uses `VITE_API_BASE_URL` and the verified `/auth/login`, `/auth/me`, and `/auth/logout` endpoints. The bearer token is held in `localStorage` so a page refresh can restore the session. This is convenient for a local prototype but exposes the token to JavaScript running on the page; it is not equivalent to an HttpOnly cookie. Do not use this as a claim of production-grade token security. Never store passwords or print tokens to logs.

Logout always clears local session state, even if the API request fails. Since the backend v1 token is stateless, clearing the client token does not necessarily revoke an already-issued JWT server-side.

## Current backend integration gap

Verified routes: `/health`, `/ready`, `/api/v1/auth/login`, `/api/v1/auth/logout`, `/api/v1/auth/me`. The dashboard summary endpoint is not registered, so the dashboard intentionally displays an unavailable state and does not render fabricated metrics.
