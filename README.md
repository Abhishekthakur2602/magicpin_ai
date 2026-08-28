# Vera bot — magicpin AI Challenge submission

## What this is
A FastAPI service implementing the 5 required endpoints (`/v1/context`,
`/v1/tick`, `/v1/reply`, `/v1/healthz`, `/v1/metadata`) plus a composer
engine that turns `(category, merchant, trigger, customer?)` into the next
Vera message: `body`, `cta`, `send_as`, `suppression_key`, `rationale`.

## Architecture
```
llm_providers.py   pluggable LLM client (Anthropic / OpenAI / Gemini / DeepSeek), temp=0
composer.py        compose() = try LLM -> fall back to rule-based templates -> validate_and_repair()
bot.py              FastAPI app: context store, tick loop, reply state machine
generate_submission.py  offline: compose() over the 30 canonical test pairs -> submission.jsonl
```

**compose() always runs every candidate — LLM or rule-based — through
`validate_and_repair()`**, which enforces the hard constraints regardless of
source: single CTA, no fabricated data, taboo-vocabulary stripping,
anti-repetition against `conversation_history`, and a guaranteed
non-empty, well-formed output. This means a misbehaving or slow LLM call
degrades to a decent deterministic message instead of crashing a tick or
producing garbage — and the bot is fully testable with **zero LLM budget**.

## Model choice
Default provider is **Anthropic Claude** (`claude-sonnet-4-6`), set via:
```
LLM_PROVIDER=anthropic
LLM_API_KEY=<key>
```
OpenAI, Gemini, DeepSeek, **Groq**, and **Cerebras** are implemented
identically — swap the env var, no code changes:
```
LLM_PROVIDER=groq
LLM_API_KEY=<your groq key>
LLM_MODEL=llama-3.3-70b-versatile   # optional override
```
```
LLM_PROVIDER=cerebras
LLM_API_KEY=<your cerebras key>
LLM_MODEL=llama-3.3-70b             # optional override
```
Groq and Cerebras are both free-tier-friendly and fast (useful against the
judge's 30s timeout), but their free models are generally weaker than
Claude/GPT-4o at following the strict JSON-only contract and nuanced tone
instructions in the system prompt — `validate_and_repair()` is the safety
net that keeps output well-formed regardless of which model is behind it.

If `LLM_PROVIDER`/`LLM_API_KEY` are unset (or the call fails/times out), the
bot uses the rule-based fallback in `composer.py`, which is what produced
the included `submission.jsonl` since this dev environment has no API key
configured. **Set a real key before judging** for full-quality LLM-composed
output — the architecture doesn't change.

**Never hardcode API keys in the source files.** Set them as environment
variables on whatever host you deploy to (Railway/Render/Fly.io all have a
"secrets"/"environment variables" panel in their dashboard).

## Decision quality / restraint
`/v1/tick` calls `should_send()` before composing anything: it skips a
trigger if its `suppression_key` was already sent, if it's expired relative
to the tick's own simulated `now` (not the server clock — this matters for
determinism), or if the merchant has 3+ consecutive unanswered Vera
messages (back off rather than spam).

## Staying inside the judge's 30s / 20-actions-per-tick budget
A single `/v1/tick` call can carry up to 20 triggers, and the whole tick has
a hard 30-second timeout — but a real LLM call (especially against a
free-tier rate limit) can easily take several seconds, so 20 sequential LLM
calls can blow both the time budget and, on constrained providers, the
token-per-minute budget. `bot.py` handles this with three techniques rather
than assuming unlimited LLM headroom:

1. **Deadline tracking** — the tick sets an internal ~24s deadline (leaving
   margin under the judge's 30s limit for network/serialization overhead).
   Before each trigger, it checks how much budget is left; once there isn't
   enough left for even one more safe LLM call, remaining triggers go
   straight to the deterministic rule-based composer instead of risking a
   timeout on the whole tick.
2. **Urgency-first ordering** — triggers are processed by descending
   `urgency` (from the trigger's own payload) rather than in arbitrary
   order, so if the LLM budget runs out partway through a large tick, it's
   the least time-sensitive triggers that degrade to rule-based, not the
   most important ones.
3. **Shrinking per-call timeout** — each LLM call is given a timeout bounded
   by *actual remaining tick budget*, not a fixed ceiling, so one slow call
   late in a tick can't itself blow past the 30s deadline.

Separately, `llm_providers.py` reduces per-call token usage (lower
`max_tokens`, and `reasoning_effort: "low"` on Groq's reasoning models like
`gpt-oss`, which otherwise burn hidden reasoning tokens against the same
rate limit as the visible output) and retries once on HTTP 429 with the
provider's own suggested wait time — but only if there's still enough
budget left to wait and retry within the caller's timeout.

**Practical recommendation for the actual judged submission:** free-tier
rate limits (e.g. Groq's default 8K tokens/minute) are workable for local
testing but genuinely tight against a 20-actions-per-tick, 30s-timeout
harness. For the real submission, either use a paid tier / higher rate
limit, or lean on the deadline-based degradation above and accept that some
lower-urgency actions in a large tick will use the rule-based composer
rather than the LLM — which is by design, not a bug, and still produces
grounded (if less nuanced) output.

## Handling thin/placeholder context
Some generated triggers carry `payload: {"placeholder": true}` with no
kind-specific facts. Rather than printing blanks or inventing numbers, the
composer falls back to real merchant-level signals — a 7-day performance
delta, a live `signals` entry, or an active offer — so every message stays
grounded in something actually present in context. This is deliberately
tested by the dataset generator and is the main failure mode called out in
the brief ("bots that pattern-match... will fail").

## /v1/reply state machine
- **Hostile message** → one apology, then `ended=True` (no more replies).
- **Hard "not interested"** → clean `end`, no argument.
- **Auto-reply detection** (common OOO phrases, or the same text repeated
  2+ times) → `hold` once, then `end` — avoids looping against a bot.
- **Explicit commitment** ("yes", "let's do it", "chalega") without a
  qualifying question in the same message → switches to **action mode**
  ("Done — drafting it now...") instead of asking another qualifying
  question, per the brief's testing guidance.
- Otherwise → one grounded follow-up question, never re-introducing Vera if
  there's prior history.

## Known limitations / tradeoffs
- In-memory context/conversation store — fine for judging a single run, not
  for a real multi-instance deployment (swap for Redis/Postgres).
- The rule-based fallback is intentionally simple per-kind template logic;
  it's a safety net, not the primary quality path. The LLM path (with a key
  set) is where category voice, tone, and CTA framing get real nuance.
- Hindi-English code-mixing in the fallback is a light heuristic string
  substitution, not real bilingual generation — the LLM path handles this
  properly via the system prompt's language-match instruction.

## Running locally
```bash
pip install -r requirements.txt
export LLM_PROVIDER=anthropic
export LLM_API_KEY=sk-ant-...
uvicorn bot:app --host 0.0.0.0 --port 8080
```

## Regenerating submission.jsonl
```bash
python3 dataset/generate_dataset.py --seed-dir dataset --out expanded
python3 generate_submission.py --dataset-dir expanded --out submission.jsonl
```

## Package layout
```
vera_bot/
├── bot.py                  FastAPI server — the 5 endpoints
├── composer.py              compose() — LLM + rule-based fallback + validation
├── llm_providers.py          pluggable LLM clients (Anthropic/OpenAI/Gemini/DeepSeek/Groq/Cerebras)
├── generate_submission.py    offline: runs compose() over the 30 canonical test pairs
├── push_and_test.py          dev helper: pushes a full expanded/ dataset into a running bot + fires a tick
├── submission.jsonl          pre-generated output (rule-based fallback — see note above)
├── requirements.txt
├── README.md                 this file
└── dataset/                  the ORIGINAL challenge seeds + generator (bundled so the
    ├── categories/            package is self-contained and reproducible without the
    ├── merchants_seed.json    original challenge zip)
    ├── customers_seed.json
    ├── triggers_seed.json
    └── generate_dataset.py    deterministic (fixed seed) — regenerate the full 50/200/100
                                 dataset + 30 test pairs any time with the command above
```