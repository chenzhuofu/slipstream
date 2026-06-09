"""Async LLM judge for the speculative summary pipeline.

Given a summary handover and a list of speculative agent actions that ran while the
summary was in flight, the judge emits a single integer 0–10 indicating whether the
speculative actions are consistent with the summary. The speculative actions are
then either carried into the new session (score ≥ threshold) or discarded.

Two variants share the same two-criterion semantics (alignment + non-contradiction):
  * ``AsyncSpecJudgeV2Fast`` — guided-decoding single-integer output, thinking
    disabled. Minimum critical-path latency.
  * ``AsyncSpecJudgeV2Slow`` — JSON output with per-criterion scores and a short
    reasoning string; model thinking enabled. Useful for debugging why a summary
    was rejected or for offline analysis.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Optional

_MAX_RETRIES = 3

# Match FoldAgent V2: strip `<think>...</think>` or `<seed:think>...</seed:think>`
# inline blocks from the summary handover before passing it to the judge. The handover
# is what the next session will see, and judging should be about that parsed text,
# not the model's internal deliberation.
_THINKING_RE = re.compile(r"<(?:seed:)?think>.*?</(?:seed:)?think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    return _THINKING_RE.sub("", text).strip()

# V2-Fast judge prompt: two-criterion evaluation, single-integer output via
# guided decoding. Shares criteria, per-score anchors, and important-context
# notes with the slow variant so scoring is directly comparable — the only
# difference is that the fast judge emits just the final integer (thinking off).
_JUDGE_PROMPT_V2_FAST = """You are judging whether a set of speculative agent actions are consistent with a summary handover.

The summary describes work already done in a prior session and what remains pending. The speculative actions ran in parallel with the summary's generation — they may or may not align with the plan.

Evaluate on TWO criteria:

1. ALIGNMENT (0-10): Do the speculative actions advance PENDING items from the summary without wandering into unrelated areas?
   - 10: every action directly targets a PENDING item
   - 7-9: actions clearly advance PENDING work, minor side-exploration allowed
   - 4-6: loose relation to PENDING; mostly-neutral exploration
   - 1-3: actions wander into areas the summary did not flag as pending
   - 0: actions pursue something totally unrelated

2. NON-CONTRADICTION (0-10): Do the speculative actions avoid undoing, redoing, or contradicting COMPLETED / CHANGES items?
   - 10: zero contradictions; nothing re-runs work already listed as done
   - 7-9: possibly minor redundancy with completed items
   - 4-6: repeats some completed exploration but doesn't destroy progress
   - 1-3: contradicts or re-opens completed work
   - 0: destroys, reverts, or clearly re-does something already marked DONE

Overall score = min(alignment, non_contradiction).

Important context for scoring:
  * Read-only exploration (grep/find/cat/head/ls/python -c probe) NEVER destroys
    state. If CHANGES is empty / no file writes happened, NON-CONTRADICTION
    should be >= 8 — read-only work cannot contradict completed changes.
  * If PENDING is broadly worded (e.g. "find where X is processed"), any search
    that narrows in on X counts as alignment — not a tangent.
  * REPAIR OF BROKEN CHANGES is alignment, not contradiction. If the summary's
    CHANGES / CURRENT_STATE explicitly flags the completed edits as "broken",
    "corrupted", "syntax error", "indentation error", or "failed restoration",
    then spec actions that `git checkout` the file or re-edit it to fix the
    breakage are REPAIR, not reversion. Give NON-CONTRADICTION >= 8.
  * Multiple sed/edit commands against a file listed in CHANGES look like
    redoing, but if the commands target a bug in the agent's OWN recent edits
    (rather than destroying a completed good fix), that is alignment.

Respond with only a single integer between 0 and 10.

---
SUMMARY HANDOVER:
{summary_handover}
---
SPECULATIVE ACTIONS (in execution order):
{spec_actions}
---

Score (0-10):"""


def _format_spec_actions_v2(spec_actions: list[dict], truncate_thinking: int = 1500) -> str:
    """Render spec actions for the judge: tool command + the agent's thinking for
    that step. Observations are intentionally omitted — the judge evaluates whether
    the agent's INTENT was aligned with the summary, not whether the command worked.

    This matches FoldAgent's ``_format_spec_actions_v2`` so the score scale stays
    comparable across projects.

    Each entry is expected to have:
      * ``command`` — shell command string (truncated to 500 chars)
      * ``thinking`` — reasoning_content captured from the assistant message
    """
    lines: list[str] = []
    for i, a in enumerate(spec_actions, 1):
        cmd = str(a.get("command", ""))[:500]
        lines.append(f"[{i}] $ {cmd}")
        thinking = str(a.get("thinking", "")).strip()
        if thinking:
            if len(thinking) > truncate_thinking:
                thinking = thinking[:truncate_thinking] + "…"
            lines.append(f"    Thinking: {thinking}")
    return "\n".join(lines) if lines else "(no speculative actions)"


def _parse_response_v2_fast(content: str) -> tuple[Optional[int], str]:
    """Parse a single-integer judge response. Returns (score, reasoning)."""
    m = re.search(r"\b(10|[0-9])\b", content.strip())
    if not m:
        return None, f"unparseable judge response: {content!r}"
    return int(m.group(1)), ""


class AsyncSpecJudgeV2Fast:
    """Single-score runtime judge: forced single-integer 0-10 via guided decoding,
    model thinking disabled. Minimizes judge latency on the critical path.

    Preserves the V2 two-criterion semantics: alignment + non-contradiction.
    """

    def __init__(
        self,
        model: str = "ByteDance-Seed/Seed-OSS-36B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 32,
        tokenizer=None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model
        self.tokenizer = tokenizer
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("JUDGE_API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("JUDGE_BASE_URL", None),
        )
        self.semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(
        self,
        summary_handover: str,
        spec_actions: list[dict],
    ) -> tuple[Optional[int], str, dict]:
        prompt = _JUDGE_PROMPT_V2_FAST.format(
            summary_handover=_strip_thinking(summary_handover),
            spec_actions=_format_spec_actions_v2(spec_actions),
        )

        last_err = ""
        for attempt in range(_MAX_RETRIES):
            async with self.semaphore:
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=8,
                        extra_body={
                            "chat_template_kwargs": {"thinking_budget": 0, "enable_thinking": False},
                            "guided_choice": [str(i) for i in range(11)],
                        },
                    )
                    usage = resp.usage
                    msg = resp.choices[0].message
                    # vLLM with --reasoning-parser exposes the parsed thinking output under
                    # different keys depending on version: as a Pydantic attribute
                    # `reasoning_content`, OR (newer SDK + seed_oss parser) as
                    # `model_extra["reasoning"]`. Check all three before giving up.
                    extra = getattr(msg, "model_extra", None) or {}
                    content = (
                        msg.content
                        or getattr(msg, "reasoning_content", None)
                        or extra.get("reasoning")
                        or extra.get("reasoning_content")
                    )
                    if not content:
                        raise ValueError("model returned empty content and reasoning_content")
                    score, reasoning = _parse_response_v2_fast(content)
                    reasoning_content = getattr(msg, "reasoning_content", None) or ""
                    if self.tokenizer is not None and reasoning_content:
                        thinking_tokens = len(
                            self.tokenizer.encode(reasoning_content, add_special_tokens=False)
                        )
                    else:
                        thinking_tokens = 0
                    token_usage = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                        "thinking_tokens": thinking_tokens,
                    }
                    return score, reasoning, token_usage
                except Exception as e:
                    last_err = str(e)
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))

        return None, f"parse error after {_MAX_RETRIES} attempts: {last_err}", {}


# ── Slow (thinking-enabled) judge: per-criterion breakdown + reasoning ───────

# Same two-criterion semantics as V2-Fast, but produces a structured JSON blob so
# we can see how the judge scored each criterion and why. Intended for debugging
# and offline analysis — not the hot path.
_JUDGE_PROMPT_V2_SLOW = """You are judging whether a set of speculative agent actions are consistent with a summary handover.

The summary describes work already done in a prior session and what remains pending. The speculative actions ran in parallel with the summary's generation — they may or may not align with the plan.

Evaluate on TWO criteria:

1. ALIGNMENT (0-10): Do the speculative actions advance PENDING items from the summary without wandering into unrelated areas?
   - 10: every action directly targets a PENDING item
   - 7-9: actions clearly advance PENDING work, minor side-exploration allowed
   - 4-6: loose relation to PENDING; mostly-neutral exploration
   - 1-3: actions wander into areas the summary did not flag as pending
   - 0: actions pursue something totally unrelated

2. NON-CONTRADICTION (0-10): Do the speculative actions avoid undoing, redoing, or contradicting COMPLETED / CHANGES items?
   - 10: zero contradictions; nothing re-runs work already listed as done
   - 7-9: possibly minor redundancy with completed items
   - 4-6: repeats some completed exploration but doesn't destroy progress
   - 1-3: contradicts or re-opens completed work
   - 0: destroys, reverts, or clearly re-does something already marked DONE

Overall score = min(alignment, non_contradiction).

Important context for scoring:
  * Read-only exploration (grep/find/cat/head/ls/python -c probe) NEVER destroys
    state. If CHANGES is empty / no file writes happened, NON-CONTRADICTION
    should be >= 8 — read-only work cannot contradict completed changes.
  * If PENDING is broadly worded (e.g. "find where X is processed"), any search
    that narrows in on X counts as alignment — not a tangent.
  * REPAIR OF BROKEN CHANGES is alignment, not contradiction. If the summary's
    CHANGES / CURRENT_STATE explicitly flags the completed edits as "broken",
    "corrupted", "syntax error", "indentation error", or "failed restoration",
    then spec actions that `git checkout` the file or re-edit it to fix the
    breakage are REPAIR, not reversion. Give NON-CONTRADICTION >= 8.
  * Multiple sed/edit commands against a file listed in CHANGES look like
    redoing, but if the commands target a bug in the agent's OWN recent edits
    (rather than destroying a completed good fix), that is alignment.

Respond with ONLY this JSON (no markdown fences, no commentary):
{{"alignment": <int 0-10>, "non_contradiction": <int 0-10>, "score": <int 0-10>, "reasoning": "<one or two sentences — what aligns, what contradicts, be specific>"}}

---
SUMMARY HANDOVER:
{summary_handover}
---
SPECULATIVE ACTIONS (in execution order):
{spec_actions}
---
"""


def _parse_response_v2_slow(content: str) -> tuple[Optional[int], str]:
    """Parse JSON {alignment, non_contradiction, score, reasoning}.

    Returns (score, reasoning). score is ``min(alignment, non_contradiction)`` —
    we ignore the model's own ``score`` field to enforce the rubric.
    """
    text = content.strip()
    # Strip optional code fences the model may add despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # Fall back to the first {...} block in case the model prefixed with prose.
    if not text.startswith("{"):
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        parsed = json.loads(text)
    except Exception as e:
        return None, f"unparseable judge JSON: {e}: {content[:200]!r}"
    try:
        a = int(parsed.get("alignment", 0))
        n = int(parsed.get("non_contradiction", 0))
        score = min(a, n)
        reasoning = str(parsed.get("reasoning", "")).strip()
        if reasoning:
            reasoning = f"alignment={a} / non_contradiction={n} — {reasoning}"
        else:
            reasoning = f"alignment={a} / non_contradiction={n}"
        return score, reasoning
    except Exception as e:
        return None, f"bad judge JSON structure: {e}: {parsed!r}"


class AsyncSpecJudgeV2Slow:
    """Thinking-enabled debug judge. Emits JSON with per-criterion scores and a
    one-sentence reasoning. Model thinking is NOT disabled, so latency is higher.

    Preserves the V2 two-criterion semantics so the score is directly comparable
    to ``AsyncSpecJudgeV2Fast``.
    """

    def __init__(
        self,
        model: str = "ByteDance-Seed/Seed-OSS-36B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        concurrency: int = 32,
        tokenizer=None,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package required: pip install openai")

        self.model = model
        self.tokenizer = tokenizer
        self.client = AsyncOpenAI(
            api_key=api_key or os.getenv("JUDGE_API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("JUDGE_BASE_URL", None),
        )
        self.semaphore = asyncio.Semaphore(concurrency)

    async def judge_one(
        self,
        summary_handover: str,
        spec_actions: list[dict],
    ) -> tuple[Optional[int], str, dict]:
        prompt = _JUDGE_PROMPT_V2_SLOW.format(
            summary_handover=_strip_thinking(summary_handover),
            spec_actions=_format_spec_actions_v2(spec_actions),
        )

        last_err = ""
        for attempt in range(_MAX_RETRIES):
            async with self.semaphore:
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=4096,
                    )
                    usage = resp.usage
                    msg = resp.choices[0].message
                    content = msg.content
                    if not content:
                        raise ValueError("model returned empty content")
                    score, reasoning = _parse_response_v2_slow(content)
                    if score is None:
                        raise ValueError(reasoning)
                    reasoning_content = getattr(msg, "reasoning_content", None) or ""
                    if self.tokenizer is not None and reasoning_content:
                        thinking_tokens = len(
                            self.tokenizer.encode(reasoning_content, add_special_tokens=False)
                        )
                    else:
                        thinking_tokens = 0
                    token_usage = {
                        "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                        "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                        "thinking_tokens": thinking_tokens,
                    }
                    return score, reasoning, token_usage
                except Exception as e:
                    last_err = str(e)
                    if attempt < _MAX_RETRIES - 1:
                        await asyncio.sleep(1.0 * (attempt + 1))

        return None, f"parse error after {_MAX_RETRIES} attempts: {last_err}", {}


Judge = AsyncSpecJudgeV2Fast | AsyncSpecJudgeV2Slow


def build_judge(cfg, tokenizer=None) -> Judge | None:
    """Return a judge instance, or None if judging is disabled.

    Judging is disabled when either `speculation_threshold` is None (always
    accept speculative actions) or `judge_model` is empty.

    Selects between the fast (default) and slow-thinking variants based on
    ``cfg.judge_thinking``. The slow variant surfaces a reasoning string that
    gets persisted into step_statistics and the summary message's ``extra``.
    """
    if getattr(cfg, "speculation_threshold", None) is None:
        return None
    if not getattr(cfg, "judge_model", ""):
        return None
    cls = AsyncSpecJudgeV2Slow if getattr(cfg, "judge_thinking", False) else AsyncSpecJudgeV2Fast
    return cls(
        model=cfg.judge_model,
        base_url=cfg.judge_base_url or None,
        api_key=cfg.judge_api_key or None,
        tokenizer=tokenizer,
    )
