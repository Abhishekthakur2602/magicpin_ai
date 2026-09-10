"""
Vera bot — FastAPI server implementing the magicpin AI Challenge contract:

    POST /v1/context     push merchant/customer/trigger/category context (idempotent by version)
    POST /v1/tick        given available trigger ids, decide what to send and compose it
    POST /v1/reply       handle an inbound merchant/customer message in an existing conversation
    GET  /v1/healthz     liveness
    GET  /v1/metadata    bot metadata

Run:
    uvicorn bot:app --host 0.0.0.0 --port 8080

Config (env vars, all optional — bot works with zero LLM budget via the
rule-based fallback in composer.py):
    LLM_PROVIDER = anthropic | openai | gemini | deepseek | none
    LLM_API_KEY  = <key>
    LLM_MODEL    = <override>

CHANGELOG (this revision):
  - /v1/tick now dedupes resolved triggers per merchant_id, keeping only the
    single highest-urgency trigger per merchant per tick. The brief is
    explicit that "strong bots ... choose the one signal that should drive
    the next message" — sending two separate messages to the same merchant
    in one tick is the message-level version of the same mistake, even if
    each individual message is otherwise fine.
  - Added an opt-in, env-gated /v1/_dev/reset endpoint purely for local
    testing convenience: SENT_SUPPRESSION_KEYS and CONTEXT_STORE are
    in-memory and persist for the life of the uvicorn process, so running
    judge_simulator.py multiple times against an already-running bot without
    restarting it will show fewer and fewer actions each run (previously-
    sent suppression keys carry over) — this looks like a regression but is
    actually correct anti-spam behavior applied across what should have been
    independent test runs. Restarting the process (or hitting this endpoint
    when enabled) gives a clean slate for local dev. Disabled by default;
    enable with VERA_DEV_RESET=1. The real harness runs one continuous
    session, so this is never needed there.
"""

from __future__ import annotations

import re
import time
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from composer import compose, should_send, g

app = FastAPI(title="Vera — magicpin AI Challenge bot")

START_TIME = time.time()

# ---------------------------------------------------------------------------
# in-memory stores (swap for redis/postgres in production; fine for judging)
# ---------------------------------------------------------------------------

CONTEXT_STORE: dict[str, dict] = {}          # key: f"{scope}:{context_id}" -> {"version":.., "payload":..}
SENT_SUPPRESSION_KEYS: set[str] = set()      # keys we've already sent for, this run
CONVERSATIONS: dict[str, dict] = {}          # conversation_id -> state


def _ckey(scope: str, context_id: str) -> str:
    return f"{scope}:{context_id}"


def get_context(scope: str, context_id: str) -> Optional[dict]:
    rec = CONTEXT_STORE.get(_ckey(scope, context_id))
    return rec["payload"] if rec else None


def get_category_for_merchant(merchant: dict) -> Optional[dict]:
    slug = merchant.get("category_slug") if merchant else None
    if not slug:
        return None
    return get_context("category", slug)


# ---------------------------------------------------------------------------
# schemas
# ---------------------------------------------------------------------------

class ContextPush(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict
    delivered_at: Optional[str] = None


class TickRequest(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = Field(default_factory=list)


class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    from_role: str
    message: str
    received_at: Optional[str] = None
    turn_number: int = 1


# ---------------------------------------------------------------------------
# /v1/context
# ---------------------------------------------------------------------------

VALID_SCOPES = {"merchant", "customer", "trigger", "category"}


@app.post("/v1/context")
def post_context(req: ContextPush):
    if req.scope not in VALID_SCOPES:
        raise HTTPException(status_code=400, detail=f"invalid scope '{req.scope}'")

    key = _ckey(req.scope, req.context_id)
    existing = CONTEXT_STORE.get(key)

    if existing and existing["version"] >= req.version:
        # idempotent no-op: same or older version
        return {
            "accepted": True,
            "ack_id": f"ack_{key}_{existing['version']}",
            "stored_at": existing["stored_at"],
        }

    stored_at = datetime.now(timezone.utc).isoformat()
    CONTEXT_STORE[key] = {"version": req.version, "payload": req.payload, "stored_at": stored_at}
    return {"accepted": True, "ack_id": f"ack_{key}_{req.version}", "stored_at": stored_at}


# ---------------------------------------------------------------------------
# /v1/tick
# ---------------------------------------------------------------------------

@app.post("/v1/tick")
def post_tick(req: TickRequest):
    actions = []

    # judge_simulator.py's BotClient.tick() opens a 30s request timeout
    # (see `self._request("POST", "/v1/tick", 30, ...)`), and the challenge
    # brief documents the same 30s harness timeout. Budget close to that,
    # leaving a safety margin for building/serializing the response and the
    # network round-trip — not an overly-conservative cap that would starve
    # later (still-urgent) triggers in a batch into the weaker rule-based
    # fallback path unnecessarily. Override via env var if your real
    # harness's timeout differs.
    tick_budget_s = float(os.environ.get("TICK_BUDGET_SECONDS", "25.0"))
    TICK_DEADLINE = time.monotonic() + tick_budget_s
    MIN_BUDGET_FOR_LLM_CALL = 3.0  # don't start an LLM call with less runway than this
    MAX_ACTIONS_PER_TICK = 20      # hard cap per the challenge contract ("20 actions/tick")

    # Resolve all valid (trigger, merchant, category, customer) tuples first,
    # then process highest-urgency triggers first — if we run out of time
    # budget or hit the action cap partway through, the triggers that
    # matter most for decision quality have already gotten the (slower,
    # better) LLM path, and the rest simply aren't actioned this tick
    # instead of the tick failing outright.
    resolved = []
    for trigger_id in req.available_triggers:
        trigger = get_context("trigger", trigger_id)
        if trigger is None:
            continue
        merchant_id = trigger.get("merchant_id")
        merchant = get_context("merchant", merchant_id) if merchant_id else None
        if merchant is None:
            continue
        category = get_category_for_merchant(merchant)
        if category is None:
            continue
        customer_id = trigger.get("customer_id")
        customer = get_context("customer", customer_id) if customer_id else None
        if not should_send(trigger, merchant, SENT_SUPPRESSION_KEYS, now=req.now):
            continue
        resolved.append((trigger_id, trigger, merchant, category, customer))

    resolved.sort(key=lambda t: t[1].get("urgency", 0), reverse=True)

    # Dedupe to one trigger per merchant per tick: after sorting by urgency,
    # keep only the first (highest-urgency) trigger we see for each
    # merchant_id and drop the rest. Without this, two independently-firing
    # triggers for the same merchant in one batch each produce their own
    # "send" action — i.e. the merchant gets two separate WhatsApp messages
    # in the same tick. The brief is explicit that strong bots "choose the
    # one signal that should drive the next message" rather than acting on
    # every available fact; sending twice in one tick is that same failure
    # mode at the message level, not just within a single message's body.
    seen_merchants: set[str] = set()
    deduped = []
    for item in resolved:
        merchant_id = item[2].get("merchant_id")
        if merchant_id in seen_merchants:
            continue
        seen_merchants.add(merchant_id)
        deduped.append(item)
    resolved = deduped

    for trigger_id, trigger, merchant, category, customer in resolved:
        if len(actions) >= MAX_ACTIONS_PER_TICK:
            break  # contract cap: never return more than 20 actions/tick

        remaining = TICK_DEADLINE - time.monotonic()
        allow_llm = remaining > MIN_BUDGET_FOR_LLM_CALL
        # bound the network call itself to what's actually left, minus a
        # safety margin for building the response — never hand the LLM
        # provider more time than we can actually afford to wait
        call_timeout = max(1.0, remaining - 1.5) if allow_llm else None
        result = compose(category, merchant, trigger, customer,
                          allow_llm=allow_llm, timeout=call_timeout)

        SENT_SUPPRESSION_KEYS.add(result["suppression_key"])

        actions.append({
            "trigger_id": trigger_id,
            "merchant_id": merchant.get("merchant_id"),
            "customer_id": trigger.get("customer_id"),
            "action": "send",
            "body": result["body"],
            "cta": result["cta"],
            "send_as": result["send_as"],
            "suppression_key": result["suppression_key"],
            "rationale": result["rationale"],
        })

        if time.monotonic() >= TICK_DEADLINE:
            break  # remaining triggers, if any, simply aren't actioned this tick

    return {"actions": actions}


# ---------------------------------------------------------------------------
# /v1/reply — conversation state machine
# ---------------------------------------------------------------------------

HOSTILE_PATTERNS = re.compile(
    r"\b(stop messaging|spam|f\*+ck|fuck off|shut up|harass|scam|leave me alone)\b",
    re.IGNORECASE,
)
HARD_NO_PATTERNS = re.compile(r"\b(not interested|no thanks|no thank you|don'?t contact)\b", re.IGNORECASE)
COMMIT_PATTERNS = re.compile(
    r"\b(yes|ok(ay)?|sure|let'?s do it|go ahead|chalega|haan|proceed|confirm(ed)?)\b",
    re.IGNORECASE,
)
QUALIFYING_PATTERNS = re.compile(r"\b(would you|do you|can you tell|what if|how about)\b", re.IGNORECASE)

APOLOGY = "Sorry to bother you — I'll stop these messages. You can always reach out if you need anything."


def _conv_state(conv_id: str) -> dict:
    if conv_id not in CONVERSATIONS:
        CONVERSATIONS[conv_id] = {"turns": [], "auto_reply_streak": 0, "ended": False}
    return CONVERSATIONS[conv_id]


def _looks_like_auto_reply(message: str, state: dict) -> bool:
    common_auto = [
        "will get back to you", "currently unavailable", "out of office",
        "thank you for your message", "we will reply shortly",
    ]
    low = message.lower()
    if any(p in low for p in common_auto):
        return True
    # same exact text seen 2+ times already in this conversation from this role
    repeats = sum(1 for t in state["turns"] if t.get("message", "").strip() == message.strip())
    return repeats >= 2


@app.post("/v1/reply")
def post_reply(req: ReplyRequest):
    state = _conv_state(req.conversation_id)
    state["turns"].append({"role": req.from_role, "message": req.message, "turn": req.turn_number})

    if state["ended"]:
        return {"action": "end", "body": ""}

    message = req.message.strip()

    # 1. hostile -> apologize once and end
    if HOSTILE_PATTERNS.search(message):
        state["ended"] = True
        return {"action": "send", "body": APOLOGY, "cta": "none", "send_as": "vera"}

    # 2. explicit hard no -> end gracefully, no apology needed
    if HARD_NO_PATTERNS.search(message):
        state["ended"] = True
        return {"action": "end", "body": ""}

    # 3. auto-reply detection -> don't loop forever
    if _looks_like_auto_reply(message, state):
        state["auto_reply_streak"] += 1
        if state["auto_reply_streak"] >= 2:
            state["ended"] = True
            return {"action": "end", "body": ""}
        return {"action": "hold", "body": ""}
    else:
        state["auto_reply_streak"] = 0

    # 4. explicit commitment -> switch to ACTION mode, not another qualifying question
    if COMMIT_PATTERNS.search(message) and not QUALIFYING_PATTERNS.search(message):
        merchant = get_context("merchant", req.merchant_id) or {}
        category = get_category_for_merchant(merchant) or {}
        offer = None
        for o in g(merchant, "offers", default=[]) or []:
            if o.get("status") == "active":
                offer = o
                break
        name_bit = f" for {offer['title']}" if offer else ""
        body = f"Done{name_bit} — drafting it now and I'll send it here for a final check."
        return {"action": "send", "body": body, "cta": "none", "send_as": "vera"}

    # 5. default: acknowledge and move the conversation forward with one clear next ask
    merchant = get_context("merchant", req.merchant_id) or {}
    fname = g(merchant, "identity", "owner_first_name") or "there"
    body = f"Got it, {fname}. What would help most right now — should I put together a quick draft?"
    return {"action": "send", "body": body, "cta": "open_ended", "send_as": "vera"}


# ---------------------------------------------------------------------------
# /v1/healthz, /v1/metadata
# ---------------------------------------------------------------------------

@app.get("/v1/healthz")
def healthz():
    return {"status": "ok", "uptime_s": round(time.time() - START_TIME, 1)}


@app.get("/v1/metadata")
def metadata():
    from llm_providers import get_provider
    provider = get_provider()
    return {
        "name": "vera-bot",
        "version": "1.0.0",
        "llm_provider": provider.name() if provider else "rule-based-fallback",
        "endpoints": ["/v1/context", "/v1/tick", "/v1/reply", "/v1/healthz", "/v1/metadata"],
        "deterministic": True,
    }


# ---------------------------------------------------------------------------
# dev-only reset endpoint (disabled unless VERA_DEV_RESET=1)
# ---------------------------------------------------------------------------
# Purely a local-testing convenience: running judge_simulator.py repeatedly
# against a still-live bot process accumulates SENT_SUPPRESSION_KEYS across
# runs, since it's an in-memory set that only clears on process restart. The
# real harness runs one continuous session and never needs this — never
# enable VERA_DEV_RESET in anything resembling the graded deployment.

if os.environ.get("VERA_DEV_RESET") == "1":
    @app.post("/v1/_dev/reset")
    def dev_reset():
        CONTEXT_STORE.clear()
        SENT_SUPPRESSION_KEYS.clear()
        CONVERSATIONS.clear()
        return {"reset": True}
    