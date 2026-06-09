"""Agent that summarizes its context when it grows too large, then continues in a fresh session.

Supports two execution modes for the summary step:

* **Sync** (default, `enable_speculation=False`): when context crosses the trigger, the
  agent blocks, generates the summary, resets the session, and continues. Identical to
  the original design.
* **Async / speculative** (`enable_speculation=True`): the summary is fired into a
  per-agent background thread while the main loop keeps doing agent + action steps on
  the *old* session. Those steps are buffered. When the summary arrives, an optional
  judge scores whether the buffered actions are consistent with the summary.
    - On *accept* (score ≥ threshold or judge disabled): start a fresh session with the
      async summary as handover and include a compact carry-over of the buffered work.
    - On *reject* (score < threshold): discard the async summary, keep the buffered
      actions in the active context, and run a **synchronous fallback summary** over the
      combined (pre-spec + spec) context. The spec work is never thrown away — only the
      misaligned summary is.

This mirrors FoldAgent's speculative summary pattern; see `async_summary_implementation.md`.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import litellm

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.judge import build_judge
from minisweagent.agents.tool_spec import TOOL_PROMPT, code_summary_tool, convert_tools_to_description, execute_bash_tool
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.retry import retry

_retry_logger = logging.getLogger("summarizing_agent")

SUMMARY_TRIGGER_TEMPLATE = (
    "The context is full. Your task will be delegated to another agent. "
    "Call the summary function now with a comprehensive handover of your progress.\n\n"
    "Here is the ground-truth state of the working tree right now "
    "(this will be included verbatim in the handover — use it to make your CHANGES "
    "field concrete and accurate; do NOT paste this diff into any summary parameter, "
    "the literal blocks below are already preserved across the handover):\n\n"
    "<git_status>\n{git_status}\n</git_status>\n\n"
    "<git_diff>\n{git_diff}\n</git_diff>\n\n"
    "Guidance for writing PENDING:\n"
    "* If the git diff already contains a plausible fix for the task, PENDING should be exactly: "
    "'Submit the patch with `git diff > patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
    "cat patch.txt`'. Do NOT list 'verify the fix', 'run tests', or 'check edge cases' — those are "
    "not required and usually fail due to missing test infrastructure in the sandbox.\n"
    "* If the git diff is empty or incomplete, PENDING should name the specific file/function to edit "
    "next, not an exploration step."
)

_SUMMARY_PARAMS = ["user_context", "completed", "pending", "current_state",
                   "code_state", "tests", "changes", "deps"]

CONTINUATION_TEMPLATE = (
    "Handover from a prior session of yours:\n\n"
    "USER_CONTEXT: {user_context}\n"
    "COMPLETED: {completed}\n"
    "PENDING: {pending}\n"
    "CURRENT_STATE: {current_state}\n"
    "CODE_STATE: {code_state}\n"
    "TESTS: {tests}\n"
    "CHANGES: {changes}\n"
    "DEPS: {deps}\n\n"
    "Working-tree snapshot at handover time (the summary's CHANGES "
    "is derived from exactly this diff; untracked files like /testbed/reproduce_issue.py "
    "appear as new-file diffs with full content via `git add -N`):\n\n"
    "<git_status>\n{git_status}\n</git_status>\n\n"
    "<git_diff>\n{git_diff}\n</git_diff>\n\n"
    "{speculative_carryover}"
    "IMPORTANT — continuing from a prior session, not starting over:\n"
    "* Files in CODE_STATE have already been examined; commands in COMPLETED have already run; edits\n"
    "  in CHANGES are already on disk. Do NOT re-`cat`/`ls`/re-grep them, do NOT re-run those commands.\n"
    "* The <git_diff> above IS the source-file state at handover. If an async carry-over block follows\n"
    "  it, the current diff is the later carry-over snapshot. Read that block instead of reconstructing\n"
    "  state from old commands, and do NOT run `git -C /testbed diff` or `git status` to verify it.\n"
    "* If a `sed -i` in the carry-over failed with bash syntax errors (parens, quotes, special chars),\n"
    "  do NOT recover by writing a Python script to a `.py` file. Use either: (a) `sed` with a different\n"
    "  delimiter like `sed -i 's|OLD|NEW|'`, or (b) an inline `python -c \"p='/testbed/x.py'; s=open(p).read();\n"
    "  open(p,'w').write(s.replace('OLD','NEW'))\"` one-liner. NEVER `cat > /testbed/<anything>.py << EOF`.\n"
    "  The `python -c` body must contain ONLY ONE `replace`/`re.sub` per turn (keep each command under\n"
    "  ~50 lines / ~2000 chars) — multi-place fixes split across turns, never one compound `python -c` rewrite.\n"
    "* If broken edits left the source file in a bad state, restore via `git -C /testbed checkout -- <path>`,\n"
    "  NEVER by writing the original content back via heredoc.\n"
    "* If the <git_diff> above shows source-file edits that plausibly address the issue, your next action\n"
    "  MUST be to submit: `git diff > patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`.\n"
    "  Tests in this sandbox often fail due to missing infra (network, plugins, test data) — a failed test\n"
    "  is NOT evidence the fix is wrong. Submit and stop spending steps on verification.\n\n"
    "Now continue working on it."
)

# Cap each git output at ~8K chars to avoid blowing out the continuation message on huge diffs.
_GIT_OUTPUT_MAX_CHARS = 8000
_SPEC_CARRYOVER_COMMAND_MAX_CHARS = 500
_SPEC_CARRYOVER_OUTPUT_MAX_CHARS = 1000
_SPEC_CARRYOVER_TOTAL_MAX_CHARS = 5000

SPECULATION_NOTICE = (
    "A background handover summary is being prepared from the context above. "
    "Until it lands, continue only with small, low-risk steps:\n"
    "* Keep your reasoning under ~120 tokens — state the goal in one short paragraph, "
    "then issue the command. No long enumeration of alternatives, multi-bullet plans, or "
    "step-by-step pre-mortems. The summary is being written in parallel; you do not need "
    "to also re-derive the situation here.\n"
    "* Use one focused command per turn.\n"
    "* Prefer targeted inspection or one in-place edit; do not start broad exploration.\n"
    "* Do not use heredoc rewrites, AST patchers, or compound `python -c` edits.\n"
    "* If the current diff already plausibly fixes the issue, submit instead of adding more checks."
)

SPEC_CARRYOVER_TEMPLATE = (
    "Async carry-over after the handover snapshot:\n"
    "The following speculative steps already ran while the summary was being prepared. "
    "Long commands and outputs are intentionally truncated; the post-spec git snapshot below is "
    "the authoritative current source state.\n\n"
    "{steps}\n\n"
    "Working-tree snapshot after speculative steps:\n\n"
    "<git_status_after_spec>\n{git_status}\n</git_status_after_spec>\n\n"
    "<git_diff_after_spec>\n{git_diff}\n</git_diff_after_spec>\n\n"
    "Keep your reasoning under ~120 tokens for the next turn — the handover above already "
    "summarizes prior progress; do not re-derive it here. State the next concrete goal in one "
    "short paragraph, then issue ONE focused command.\n\n"
)

SPEC_CARRYOVER_TEMPLATE_NO_CHANGES = (
    "Async carry-over after the handover snapshot:\n"
    "The following speculative steps already ran while the summary was being prepared. "
    "Long commands and outputs are intentionally truncated.\n\n"
    "{steps}\n\n"
    "(speculation made no working-tree changes; the handover snapshot above is still authoritative.)\n\n"
    "Keep your reasoning under ~120 tokens for the next turn — the handover above already "
    "summarizes prior progress; do not re-derive it here. State the next concrete goal in one "
    "short paragraph, then issue ONE focused command.\n\n"
)

COMMAND_LOOP_NOTICE = (
    "⚠ Command loop detected. Your last few commands have been near-identical to commands "
    "that ran earlier in this session — you are not making progress on the task. On your VERY "
    "next turn, EITHER:\n"
    "  (a) commit to a `sed -i 's/OLD/NEW/' /testbed/<path>` edit on the most likely line, OR\n"
    "  (b) submit whatever is currently on disk via "
    "`git diff > patch.txt && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt`.\n"
    "Do NOT run another exploration command (cat / grep / sed -n / python -c probe). "
    "Reading more code will not break the loop."
)

FIXED_POINT_NOTICE = (
    "⚠ Fixed-point handover detected: this handover's PENDING and CHANGES are identical to "
    "the previous one's. The previous session did not act on PENDING — it just summarized again. "
    "On your VERY first command in this session you MUST execute the PENDING action concretely "
    "(e.g. if PENDING says 'change line X to Y', run the corresponding `sed -i` for it now), "
    "or submit whatever is on disk if PENDING is too abstract. Do NOT explore."
)


def _command_signature(cmd: str) -> str:
    """Compute a normalized hash of a shell command for loop detection.

    Strips leading/trailing whitespace, collapses repeated whitespace, lowercases, and truncates
    to the first 200 chars (so cosmetic-only differences like print-statement variations don't
    masquerade as distinct commands)."""
    if not cmd:
        return ""
    norm = " ".join(cmd.strip().lower().split())[:200]
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _summary_signature(params: dict) -> str:
    """Hash a parsed summary's params for fixed-point detection.

    Compares structured fields (user_context, completed, pending, ...) — not the raw response
    text. Two summaries are 'fixed-point identical' iff every parsed parameter matches."""
    canon = json.dumps({k: (params.get(k) or "").strip() for k in _SUMMARY_PARAMS}, sort_keys=True)
    return hashlib.sha1(canon.encode()).hexdigest()[:16]


def _extract_asst_thinking(asst_msg: dict) -> str:
    """Return the assistant message's ``reasoning_content`` (thinking tokens) if any.

    litellm stores the full response under ``extra.response`` via ``response.model_dump()``;
    vLLM with a reasoning_parser puts thinking text in ``choices[0].message.reasoning_content``.
    """
    try:
        resp = (asst_msg.get("extra") or {}).get("response") or {}
        choices = resp.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return str(msg.get("reasoning_content") or "")
    except Exception:
        return ""


def _truncate_git(out: str, max_chars: int = _GIT_OUTPUT_MAX_CHARS) -> str:
    out = out.strip()
    if len(out) <= max_chars:
        return out or "(empty)"
    head = out[: max_chars // 2]
    tail = out[-max_chars // 2 :]
    elided = len(out) - max_chars
    return f"{head}\n...[{elided} chars elided]...\n{tail}"


def _truncate_middle(text: str, max_chars: int) -> str:
    text = text.strip()
    if not text:
        return "(empty)"
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[-max_chars // 2 :]
    return f"{head}\n...[{len(text) - max_chars} chars elided]...\n{tail}"


def _message_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


class SummarizingAgentConfig(AgentConfig):
    summary_context_tokens: int = 0
    """Trigger a summary when the token count of session-specific messages exceeds
    ``summary_context_tokens * context_full_trigger`` (0 = summary disabled).
    The fixed system prompt + initial task message are excluded from the count since
    they persist unchanged across sessions."""

    # ---- Async / speculative summary knobs ---------------------------------
    enable_speculation: bool = False
    """Master switch for the async summary pipeline. When False, behaviour is identical
    to the original sync implementation."""

    context_full_trigger: float = 1.0
    """Fraction of ``summary_context_tokens`` at which the summary is fired. Values below
    1.0 leave headroom for speculative agent steps while the summary computes in the
    background."""

    speculation_length: int = 0
    """Extra token budget the agent may accumulate in the active session *beyond*
    ``summary_context_tokens`` while a speculative summary is in flight. The hard cap is
    ``summary_context_tokens + speculation_length`` — once exceeded, the main loop blocks
    on the summary future."""

    speculation_step_notice: bool = True
    """Append a short transient instruction while a summary is in flight so speculative
    agent steps know they are operating in a stale, near-full context and should keep
    commands small."""

    speculation_threshold: Optional[int] = None
    """Minimum judge score (0-10) required to carry speculative actions forward into the
    new session. ``None`` => always accept (no judge is invoked)."""

    judge_model: str = ""
    """Model name for the speculation judge. Empty => judging disabled regardless of
    ``speculation_threshold``."""

    judge_base_url: str = ""
    """OpenAI-compatible base URL for the judge (fallback: ``JUDGE_BASE_URL`` env var)."""

    judge_api_key: str = ""
    """API key for the judge client (fallback: ``JUDGE_API_KEY`` env var)."""

    judge_thinking: bool = False
    """When True, use the slow ``AsyncSpecJudgeV2Slow`` (model thinking ENABLED, JSON
    output with per-criterion scores + reasoning). When False (default), uses the fast
    ``AsyncSpecJudgeV2Fast`` (guided-decoding single integer, thinking off). The slow
    judge writes a ``judge_reasoning`` string to step_statistics so post-hoc viewers
    can show why a summary was accepted or rejected."""


class SummarizingAgent(DefaultAgent):
    def __init__(self, model, env, *, config_class=SummarizingAgentConfig, **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.sessions: list[list[dict]] = []
        self.full_history: list[dict] = []
        # Profiling: mirrors FoldAgent's step_statistics — one entry per agent/summary/action step.
        self.step_statistics: list[dict] = []
        self._iteration: int = 0
        self._run_start_time: float = time.time()

        # -- Speculative summary state (only active when enable_speculation=True) --
        # Dedicated single-worker executor so at most one summary is in flight per agent.
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"spec-summary-{id(self)}"
        )
        self._spec_future: Future | None = None
        # Buffered (assistant_msg, observation_msgs) pairs produced while speculating.
        self._spec_buffer: list[tuple[dict, list[dict]]] = []
        # Snapshot of state at the moment speculation started — used for rollback on reject.
        self._pre_spec: dict = {}
        # Judge for speculation (None = always accept).
        self._judge = build_judge(self.config)

        # -- Loop-detection state --
        # Rolling window of recent agent-command signatures. We compare last 3 vs prior 3 to
        # detect "stuck" sessions where the agent keeps running near-identical commands without
        # making progress (the dominant LimitsExceeded failure mode for weak models).
        self._recent_cmd_sigs: list[str] = []
        self._command_loop_pending: bool = False  # set True when detected; consumed at next step
        # Hash of the most recent ACCEPTED summary's parsed params, for fixed-point detection
        # across the summary boundary. Two consecutive identical hashes => agent isn't acting on
        # PENDING; we inject a warning into the next session's continuation.
        self._last_summary_sig: Optional[str] = None

    def add_messages(self, *messages: dict) -> list[dict]:
        """Append to both the active context and the cross-session full history."""
        result = super().add_messages(*messages)
        self.full_history.extend(result)
        return result

    def get_template_vars(self, **kwargs) -> dict:
        tools = execute_bash_tool()
        if self.config.summary_context_tokens > 0:
            tools += code_summary_tool()
        tool_description = TOOL_PROMPT.format(description=convert_tools_to_description(tools))
        return super().get_template_vars(tool_description=tool_description, **kwargs)

    def _context_tokens(self) -> int:
        """Count tokens in session-specific messages only, excluding the fixed
        system prompt and initial task message (self.messages[:2]) which are
        constant across sessions and thus represent baseline overhead, not
        summarizable context."""
        session_messages = self.messages[2:]
        if not session_messages:
            return 0
        return litellm.token_counter(model=self.model.config.model_name, messages=session_messages)

    def _query_raw(self, messages: list[dict] | None = None) -> dict:
        """Query the model bypassing action parsing. Needed because the summary call uses
        <function=summary>, which would be rejected by the configured <function=execute_bash> regex.

        Pass an explicit ``messages`` list to query a frozen snapshot (required when called
        from a background thread during speculation, since the main thread continues mutating
        ``self.messages``). Defaults to ``self.messages`` for the sync path.
        """
        msgs = self.messages if messages is None else messages
        for attempt in retry(logger=_retry_logger, abort_exceptions=self.model.abort_exceptions):
            with attempt:
                response = self.model._query(self.model._prepare_messages_for_api(msgs))
        cost_output = self.model._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        msg = response.choices[0].message.model_dump()
        msg["extra"] = {
            "actions": [],
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return msg

    def _capture_git_snapshot(self) -> tuple[str, str]:
        """Capture objective working-tree state to inject into the summary.

        Runs `git add -N .` once up front so any UNTRACKED files (commonly
        /testbed/reproduce_issue.py or new source files the agent created) are
        marked as intent-to-add. Two consequences:
          * `git status --short` shows them as `A` instead of `??` (cleaner block)
          * `git diff HEAD` emits them as full-content new-file diffs

        `add -N` only stages "intent to add" (no content), is non-destructive to
        working-tree files, and benignly persists in the index so the agent's own
        subsequent `git diff` calls also include the new files.

        At small ``summary_context_tokens`` (≤4k) the default 8K-char cap on each of
        status/diff means the continuation message can start a fresh session at
        ~4K tokens of pure handover, immediately tripping the next trigger. Scale
        the cap down to 3000 chars in that regime."""
        cap = 3000 if self.config.summary_context_tokens <= 4096 else _GIT_OUTPUT_MAX_CHARS
        try:
            status_out = self.env.execute({"command": "git -C /testbed add -N . && git -C /testbed status --short 2>&1"})
            diff_out = self.env.execute({"command": "git -C /testbed diff HEAD 2>&1"})
            status = _truncate_git(status_out.get("output", ""), max_chars=cap)
            diff = _truncate_git(diff_out.get("output", ""), max_chars=cap)
        except Exception as e:
            status = f"(git status unavailable: {e})"
            diff = f"(git diff unavailable: {e})"
        return status, diff

    @staticmethod
    def _parse_summary_params(content: str) -> dict:
        """Extract the 9 summary parameters from a ``<function=summary>`` blob."""
        params = {p: "" for p in _SUMMARY_PARAMS}
        fn_match = re.search(r"<function=summary>(.*?)</function>", content, re.DOTALL)
        if fn_match:
            for param_name in _SUMMARY_PARAMS:
                m = re.search(rf"<parameter={param_name}>(.*?)</parameter>", fn_match.group(1), re.DOTALL)
                if m:
                    params[param_name] = m.group(1).strip()
        else:
            params["completed"] = content.strip()
        return params

    def _build_summary_trigger_msg(self) -> tuple[dict, str, str]:
        """Render the summary-trigger user message. Returns (message, git_status, git_diff)
        so the same git snapshot can be reused in the continuation message without a second
        git call."""
        git_status, git_diff = self._capture_git_snapshot()
        content = SUMMARY_TRIGGER_TEMPLATE.format(git_status=git_status, git_diff=git_diff)
        return self.model.format_message(role="user", content=content), git_status, git_diff

    def _new_session_messages(
        self, params: dict, git_status: str, git_diff: str, speculative_carryover: str = "",
        trailing_notice: str = "",
    ) -> list[dict]:
        """Return the initial 4-message list for a fresh session after a summary:
        [system, task, placeholder_asst, continuation_user]."""
        continuation = CONTINUATION_TEMPLATE.format(
            **params,
            git_status=git_status,
            git_diff=git_diff,
            speculative_carryover=speculative_carryover,
        )
        if trailing_notice:
            continuation = continuation.rstrip() + "\n\n" + trailing_notice
        return [
            self.messages[0],
            self.messages[1],
            self.model.format_message(role="assistant", content=" "),
            self.model.format_message(role="user", content=continuation),
        ]

    # ------------------------------------------------------------------
    # Sync summary (legacy path, used when enable_speculation=False)
    # ------------------------------------------------------------------
    def _do_summary(self, fallback: bool = False) -> None:
        """Request a summary function call from the model, then start a fresh session.

        ``fallback=True`` tags the recorded summary stat entry so downstream analysis can
        distinguish sync-fallback summaries (triggered after a rejected speculation) from
        regular sync summaries.
        """
        self.sessions.append(list(self.messages))

        trigger, git_status, git_diff = self._build_summary_trigger_msg()
        self.add_messages(trigger)
        t0 = time.time()
        summary_msg = self._query_raw()
        summary_time = time.time() - t0
        self.cost += summary_msg.get("extra", {}).get("cost", 0.0)
        self.full_history.append(summary_msg)

        usage = (summary_msg.get("extra", {}).get("response") or {}).get("usage") or {}
        self.step_statistics.append({
            "type": "summary",
            "step": self._iteration,
            "time": summary_time,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "speculative_steps": 0,
            "fallback": fallback,
        })

        params = self._parse_summary_params(summary_msg.get("content") or "")
        # Fixed-point detection: identical params hash => the agent isn't acting on PENDING.
        sig = _summary_signature(params)
        is_fixed_point = self._last_summary_sig is not None and sig == self._last_summary_sig
        self._last_summary_sig = sig
        if is_fixed_point:
            trailing_notice = FIXED_POINT_NOTICE
        else:
            trailing_notice = ""

        new_msgs = self._new_session_messages(
            params, git_status, git_diff, trailing_notice=trailing_notice,
        )
        self.messages = new_msgs[:2]
        self.add_messages(*new_msgs[2:])

    # ------------------------------------------------------------------
    # Async speculative summary
    # ------------------------------------------------------------------
    def _run_summary_and_judge(
        self,
        snapshot: list[dict],
        spec_buffer_ref: list[tuple[dict, list[dict]]],
    ) -> tuple[dict, float, Optional[int], str, float, dict]:
        """Background-thread worker: generate summary against a frozen snapshot, then
        optionally invoke the async judge on whatever spec actions have accumulated so far.

        Returns (summary_msg, summary_time, judge_score, judge_reasoning, judge_time, judge_usage).
        ``judge_reasoning`` is the empty string for the fast judge.
        """
        t0 = time.time()
        summary_msg = self._query_raw(messages=snapshot)
        summary_time = time.time() - t0

        judge_score: Optional[int] = None
        judge_reasoning: str = ""
        judge_time: float = 0.0
        judge_usage: dict = {}

        if self._judge is not None and spec_buffer_ref:
            # Snapshot the buffer at judge time so further main-thread appends don't race us.
            # Match FoldAgent V2: pass (command, thinking) per action; drop observation.
            spec_snapshot = [
                {
                    "command": (asst.get("extra", {}).get("actions") or [{}])[0].get("command", ""),
                    "thinking": _extract_asst_thinking(asst),
                }
                for asst, _obs in list(spec_buffer_ref)
            ]
            t1 = time.time()
            summary_content = summary_msg.get("content") or ""
            score, reasoning, judge_usage = asyncio.run(
                self._judge.judge_one(summary_content, spec_snapshot)
            )
            judge_time = time.time() - t1
            judge_score = score
            judge_reasoning = reasoning or ""

        return summary_msg, summary_time, judge_score, judge_reasoning, judge_time, judge_usage

    def _fire_summary_async(self) -> None:
        """Kick off a background summary job. Non-blocking."""
        # Snapshot messages + trigger for the background thread's input.
        trigger, git_status, git_diff = self._build_summary_trigger_msg()
        snapshot = list(self.messages) + [trigger]
        # Record the trigger in the persistent history (but NOT in self.messages —
        # the main thread keeps operating on the pre-trigger context while spec is in flight).
        self.full_history.append(trigger)
        self.sessions.append(list(self.messages))

        # Stash the trigger-time git snapshot so the eventual continuation can include the
        # SAME diff the summarizer was promised in SUMMARY_TRIGGER_TEMPLATE ("included
        # verbatim in the handover"). Capturing again at consume time would show post-spec
        # state and break the prose/diff alignment.
        self._pre_spec = {
            "iteration": self._iteration,
            "messages_len": len(self.messages),
            "full_history_len": len(self.full_history),
            "git_status": git_status,
            "git_diff": git_diff,
        }
        self._spec_buffer = []
        self._spec_future = self._executor.submit(
            self._run_summary_and_judge, snapshot, self._spec_buffer
        )
        if self.config.speculation_step_notice:
            self.add_messages(self.model.format_message(role="user", content=SPECULATION_NOTICE))

    def _format_speculative_carryover(
        self, spec_buffer: list[tuple[dict, list[dict]]], git_status: str, git_diff: str,
        no_changes: bool = False,
    ) -> str:
        """Create a bounded, non-demonstrative record of speculative work.

        Replaying raw assistant messages after an accepted summary preserves long command
        bodies as examples for the next session. The compact block records what happened
        and gives the current diff without encouraging the model to imitate those commands.

        ``no_changes=True`` means the post-spec git snapshot is identical to the trigger-time
        snapshot already shown in the main CONTINUATION block — speculation only ran read-only
        commands (cat / grep / sed -n / python -c print). In that case, omit the duplicate
        ``<git_diff_after_spec>`` block and emit a one-liner instead, saving ~750 tokens of
        per-session continuation overhead on the dominant case.
        """
        if not spec_buffer:
            return ""

        entries: list[str] = []
        used_chars = 0
        for i, (asst, obs_messages) in enumerate(spec_buffer, start=1):
            command = (asst.get("extra", {}).get("actions") or [{}])[0].get("command", "")
            command = _truncate_middle(str(command), _SPEC_CARRYOVER_COMMAND_MAX_CHARS)
            obs_parts = []
            for obs in obs_messages:
                raw_output = (obs.get("extra") or {}).get("raw_output")
                output = raw_output if raw_output is not None else _message_text(obs)
                returncode = (obs.get("extra") or {}).get("returncode", "")
                exception_info = (obs.get("extra") or {}).get("exception_info") or ""
                obs_text = f"returncode={returncode}\n{output}"
                if exception_info:
                    obs_text = f"{exception_info}\n{obs_text}"
                obs_parts.append(obs_text)
            observation = _truncate_middle("\n\n".join(obs_parts), _SPEC_CARRYOVER_OUTPUT_MAX_CHARS)
            entry = (
                f"Speculative step {i}:\n"
                f"<command_truncated_if_long>\n{command}\n</command_truncated_if_long>\n"
                f"<observation_truncated_if_long>\n{observation}\n</observation_truncated_if_long>"
            )
            if used_chars + len(entry) > _SPEC_CARRYOVER_TOTAL_MAX_CHARS:
                remaining = len(spec_buffer) - i + 1
                entries.append(
                    f"...[{remaining} later speculative step(s) omitted from the compact carry-over; "
                    "use the post-spec git snapshot below as the source of truth.]..."
                )
                break
            entries.append(entry)
            used_chars += len(entry)

        if no_changes:
            return SPEC_CARRYOVER_TEMPLATE_NO_CHANGES.format(steps="\n\n".join(entries))
        return SPEC_CARRYOVER_TEMPLATE.format(
            steps="\n\n".join(entries),
            git_status=git_status,
            git_diff=git_diff,
        )

    def _consume_summary_future(self) -> None:
        """Drain the completed summary future and decide what to do with its output.

        * **Accept** (judge score ≥ threshold, judge disabled, or judge failed): start a fresh
          session with the async summary as handover, then compact the buffered spec pairs
          into a bounded carry-over block (or replay them if compatibility mode is enabled).
        * **Reject** (judge score < threshold): discard the async summary entirely. The
          speculative actions remain in ``self.messages`` (they were appended normally during
          ``step()``), so we invoke :py:meth:`_do_summary` synchronously to summarize the
          combined (pre-spec + spec) context. The iteration counter is NOT rolled back.
        """
        assert self._spec_future is not None
        summary_msg, summary_time, judge_score, judge_reasoning, judge_time, judge_usage = self._spec_future.result()

        usage = (summary_msg.get("extra", {}).get("response") or {}).get("usage") or {}
        if self._judge is not None:
            self.step_statistics.append({
                "type": "judge",
                "step": self._pre_spec["iteration"],
                "time": judge_time,
                "score": judge_score,
                "reasoning": judge_reasoning,
                "input_tokens": judge_usage.get("input_tokens", 0),
                "output_tokens": judge_usage.get("output_tokens", 0),
                "thinking_tokens": judge_usage.get("thinking_tokens", 0),
            })

        accept = (
            self.config.speculation_threshold is None
            or judge_score is None  # judge failed -> fall back to accepting
            or judge_score >= self.config.speculation_threshold
        )

        if accept:
            # Pay the cost of the async summary and splice it into history.
            self.cost += summary_msg.get("extra", {}).get("cost", 0.0)
            summary_msg.setdefault("extra", {})["judge_score"] = judge_score
            summary_msg["extra"]["speculative_steps"] = len(self._spec_buffer)
            if judge_reasoning:
                summary_msg["extra"]["judge_reasoning"] = judge_reasoning
            self.full_history.append(summary_msg)
            self.step_statistics.append({
                "type": "summary",
                "step": self._pre_spec["iteration"],
                "time": summary_time,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "speculative_steps": len(self._spec_buffer),
                "judge_score": judge_score,
                "fallback": False,
            })
            params = self._parse_summary_params(summary_msg.get("content") or "")
            # Fixed-point detection on the accepted summary's parsed params — same as sync path.
            sig = _summary_signature(params)
            is_fixed_point = self._last_summary_sig is not None and sig == self._last_summary_sig
            self._last_summary_sig = sig
            if is_fixed_point:
                trailing_notice = FIXED_POINT_NOTICE
            else:
                trailing_notice = ""

            # Reuse the trigger-time git snapshot stashed in _pre_spec — this is the
            # SAME diff the summarizer saw, so summary prose and continuation diff stay
            # aligned. Accepted speculative work is compacted below so the next session
            # sees the post-spec state without raw long commands becoming examples.
            speculative_carryover = ""
            compact_carryover = judge_score is None or judge_score <= self.config.speculation_threshold + 3
            if compact_carryover and self._spec_buffer:
                post_spec_git_status, post_spec_git_diff = self._capture_git_snapshot()
                # Skip the duplicate <git_diff_after_spec> block when speculation made no
                # working-tree changes (read-only cat/grep/sed-n/python-c print). On 12907-style
                # instances the post-spec snapshot is identical to the trigger-time one ~80%
                # of the time, and the duplicate eats ~750 tokens of per-session budget, which
                # shortens median session length from 5–6 to 3 agent steps.
                no_changes = (post_spec_git_status == self._pre_spec["git_status"]
                              and post_spec_git_diff == self._pre_spec["git_diff"])
                speculative_carryover = self._format_speculative_carryover(
                    self._spec_buffer, post_spec_git_status, post_spec_git_diff,
                    no_changes=no_changes,
                )
            new_msgs = self._new_session_messages(
                params,
                self._pre_spec["git_status"],
                self._pre_spec["git_diff"],
                speculative_carryover=speculative_carryover,
                trailing_notice=trailing_notice,
            )
            self.messages = new_msgs[:2]
            self.add_messages(*new_msgs[2:])
            if not compact_carryover and self._spec_buffer:
                for asst, obs in self._spec_buffer:
                    # Append without re-adding to full_history (they're already there via step()).
                    self.messages.append(asst)
                    self.messages.extend(obs)
            self._spec_future = None
            self._spec_buffer = []
            self._pre_spec = {}
            return

        # Reject: record the discarded async summary (cost + tokens still billed) AND
        # preserve it in full_history for post-hoc analysis. Then fall back to a sync
        # summary over the combined (pre-spec + spec) context.
        self.cost += summary_msg.get("extra", {}).get("cost", 0.0)
        summary_msg.setdefault("extra", {})["discarded"] = True
        summary_msg["extra"]["judge_score"] = judge_score
        summary_msg["extra"]["speculative_steps"] = len(self._spec_buffer)
        if judge_reasoning:
            summary_msg["extra"]["judge_reasoning"] = judge_reasoning
        self.full_history.append(summary_msg)
        self.step_statistics.append({
            "type": "summary",
            "step": self._pre_spec["iteration"],
            "time": summary_time,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "speculative_steps": len(self._spec_buffer),
            "judge_score": judge_score,
            "discarded": True,
        })
        # Clear spec state BEFORE the sync call so its own step entries are NOT tagged speculative.
        self._spec_future = None
        self._spec_buffer = []
        self._pre_spec = {}
        self._do_summary(fallback=True)

    # ------------------------------------------------------------------
    # Main step loop
    # ------------------------------------------------------------------
    def _record_step_stats(self, msg: dict, outputs: list[dict], agent_time: float, action_time: float) -> None:
        """Append agent + action entries for this iteration; tag as speculative if applicable."""
        is_spec = self._spec_future is not None
        usage = (msg.get("extra", {}).get("response") or {}).get("usage") or {}
        agent_entry = {
            "type": "agent",
            "step": self._iteration,
            "time": agent_time,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
        if is_spec:
            agent_entry["speculative"] = True
        self.step_statistics.append(agent_entry)

        obs_content = "\n".join(str(o.get("content", "")) for o in outputs)
        try:
            obs_tokens = litellm.token_counter(model=self.model.config.model_name, text=obs_content)
        except Exception:
            obs_tokens = len(obs_content) // 4  # fallback: rough char-to-token estimate
        action_entry = {
            "type": "action",
            "step": self._iteration,
            "time": action_time,
            "output_tokens": obs_tokens,
        }
        if is_spec:
            action_entry["speculative"] = True
        self.step_statistics.append(action_entry)

    def step(self) -> list[dict]:
        trigger_tokens = self.config.summary_context_tokens
        if trigger_tokens > 0:
            # --- Block 1: background summary arrived ---------------------
            if self._spec_future is not None and self._spec_future.done():
                self._consume_summary_future()

            tokens = self._context_tokens()
            soft_cap = trigger_tokens * self.config.context_full_trigger
            hard_cap = trigger_tokens + self.config.speculation_length

            # --- Block 2: fire summary when near soft cap ----------------
            if self._spec_future is None and tokens > soft_cap:
                if self.config.enable_speculation:
                    self._fire_summary_async()
                else:
                    self._do_summary()

            # --- Block 3: hard cap - must wait for the in-flight summary -
            if self._spec_future is not None and tokens > hard_cap:
                self._consume_summary_future()

        # COMMAND_LOOP_NOTICE is injected immediately as a
        # synthetic user turn so the model sees it before its very next decode.
        if self._command_loop_pending:
            self.add_messages(self.model.format_message(role="user", content=COMMAND_LOOP_NOTICE))
        self._command_loop_pending = False

        # --- Block 4: normal agent + action step -------------------------
        t_agent = time.time()
        msg = self.query()
        agent_time = time.time() - t_agent

        t_action = time.time()
        outputs = self.execute_actions(msg)
        action_time = time.time() - t_action

        if self._spec_future is not None:
            self._spec_buffer.append((msg, outputs))

        # --- Block 4.5: command-loop detection ---------------------------
        # Track the assistant's command signature; if 2 of the last 3 issued commands match 2
        # of the prior 3, the session is stuck. Set the flag so the NEXT step() injects a
        # synthetic warning (we can't inject after the fact; the model sees it before its next
        # decode). Self-clearing: once a warning is emitted, the rolling window is reset so we
        # don't fire again immediately on the same window.
        cmd = (msg.get("extra", {}).get("actions") or [{}])[0].get("command", "")
        sig = _command_signature(cmd)
        if sig:
            self._recent_cmd_sigs.append(sig)
            if len(self._recent_cmd_sigs) > 12:
                self._recent_cmd_sigs = self._recent_cmd_sigs[-12:]
            if len(self._recent_cmd_sigs) >= 6:
                last3 = set(self._recent_cmd_sigs[-3:])
                prior3 = set(self._recent_cmd_sigs[-6:-3])
                if len(last3 & prior3) >= 2:
                    self._command_loop_pending = True
                    self._recent_cmd_sigs = []  # reset so we don't re-fire on the same window

        self._record_step_stats(msg, outputs, agent_time, action_time)

        self._iteration += 1
        return outputs

    def serialize(self, *extra_dicts) -> dict:
        return super().serialize({"sessions": self.sessions, "full_history": self.full_history}, *extra_dicts)

    def _compute_speculation_metrics(self) -> dict:
        """Compute overlap / actual_savings diagnostics for speculative summaries.

        For each accepted speculative summary, find the concurrent spec spans (agent+action
        entries tagged ``speculative=True``) and measure how much of the summary's wall time
        was hidden behind them. Rejected (``discarded``) summaries contribute to the reject
        counter but not to ``actual_savings``, since the spec span is paid back by the sync
        fallback that replaces them.
        """
        metrics = {
            "n_spec_accepted": 0,
            "n_spec_rejected": 0,
            "n_summary_fallback": 0,
            "actual_savings": 0.0,
        }

        spec_times_by_step: dict[int, float] = {}
        for entry in self.step_statistics:
            if entry.get("speculative"):
                key = entry.get("step", -1)
                spec_times_by_step[key] = spec_times_by_step.get(key, 0.0) + entry.get("time", 0.0)

        for entry in self.step_statistics:
            if entry.get("type") != "summary":
                continue
            if entry.get("fallback"):
                metrics["n_summary_fallback"] += 1
            if entry.get("speculative_steps", 0) == 0:
                continue
            start_step = entry.get("step", 0)
            spec_span = sum(t for step, t in spec_times_by_step.items() if step >= start_step)
            if entry.get("discarded"):
                metrics["n_spec_rejected"] += 1
            else:
                metrics["n_spec_accepted"] += 1
                metrics["actual_savings"] += min(entry.get("time", 0.0), spec_span)
        return metrics

    def close(self) -> None:
        """Drain any in-flight speculation, then shut the executor down. Safe to call multiple times.

        We wait (with a generous timeout) for the running summary thread to finish before tearing
        down the executor — Future.cancel() does NOT stop a thread that is already executing, and
        if we shut down with wait=False while the worker is mid-litellm-call, the worker keeps
        running and eventually tries to schedule a callback on a closed executor → RuntimeError →
        litellm retry loop that never exits.
        """
        if self._spec_future is not None:
            if not self._spec_future.done():
                try:
                    self._spec_future.result(timeout=120)
                except Exception:
                    pass
            self._spec_future = None
        try:
            self._executor.shutdown(wait=True)
        except Exception:
            pass

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save trajectory to `path` and profiling stats to a sibling `<stem>.stats.json`."""
        data = super().save(path, *extra_dicts)
        if path:
            stats = {
                "instance_id": data.get("instance_id"),
                "total_time": time.time() - self._run_start_time,
                "n_sessions": len(self.sessions) + 1,
                "cost": self.cost,
                "n_calls": self.n_calls,
                "step_statistics": self.step_statistics,
                **self._compute_speculation_metrics(),
            }
            stats_path = path.parent / f"{Path(path.stem).stem}.stats.json"
            stats_path.write_text(json.dumps(stats, indent=2))
        return data

    def __del__(self):
        # Best-effort cleanup if the agent is garbage-collected without explicit close.
        try:
            self.close()
        except Exception:
            pass
