# Palette — Color Season Analyzer

## What this is
A web app that scans a selfie, classifies the user's color season, and sells a full color/style report behind a one-time Stripe paywall. V1 = color season only, free scan + paid unlock. V2 adds face shape as a bundled "Full Beauty Profile."

## Stack
- Frontend: Next.js (React), deployed on Vercel
- Backend: FastAPI (Python)
- Database: Postgres (Supabase or Neon)
- Payments: Stripe Checkout, one-time payment mode (no subscriptions)
- AI: Claude Haiku 4.5 via the Anthropic API (model string `claude-haiku-4-5-20251001`) — used ONLY for generating the short personalized write-up paragraph

## Critical rule — read this before touching the classifier
Face detection and season/shape classification are **deterministic Python, not an LLM call.** Use MediaPipe or OpenCV for face detection and skin-tone sampling, convert to HSV or Lab color space, then run a rule-based classifier against documented color theory (warm/cool, light/deep, clear/muted → one of four seasons). Do NOT implement the classification itself as a prompt to an LLM — same photo must always produce the same season, and it should cost nothing per free scan.

The only place an LLM call belongs in this app is turning an already-computed result (season name + palette data) into a short, warm paragraph of text. That's it.

## Conventions
- Keep classifier logic in small, testable functions — write unit tests with known photo → expected season pairs as you go
- Don't retain uploaded selfies after processing — extract what's needed (season, swatch data), discard the image immediately
- All secrets (ANTHROPIC_API_KEY, STRIPE_SECRET_KEY, DATABASE_URL) live in environment variables — never hardcoded, never committed
- Mobile-first responsive design throughout — assume nearly all traffic arrives on a phone from a TikTok link
- Log basic events (scan started, scan completed, paywall viewed, purchase completed) to a simple Postgres table — this is the whole analytics setup for v1, no need for a dedicated analytics product

## Current milestone (v1)
Free scan → season result (name + a few example swatches + short AI paragraph) → Stripe paywall ($2.99–$3.99 one-time) → paid result (full ~20-30 color palette organized best/good/avoid, makeup shade guidance, jewelry metal tone, personalized AI paragraph, color-family shopping guidance).

## Not yet in scope (v2 — don't build ahead of need)
Face shape detection, bundle paywall, affiliate product recommendations, referral/share mechanic, user accounts.
