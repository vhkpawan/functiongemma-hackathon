# RULES.md — FunctionGemma Hackathon (Cursor Guardrails)

You are an AI coding assistant (Cursor). You MUST follow these rules strictly.

If any request conflicts with these rules, DO NOT implement it. Instead, explain the conflict and propose a compliant alternative.

---

## 0) Primary Goal
Make a valid hackathon submission by improving the hybrid routing logic while keeping the repo fully compatible with the provided benchmark and submission pipeline.

Stability + compatibility > refactors.

---

## 1) Allowed Edit Scope (STRICT)
You may ONLY modify:

- `main.py` — **ONLY the internal logic** of `generate_hybrid()`.

You MAY:
- Add small helper functions **inside `main.py`** (below or above `generate_hybrid()`).
- Add lightweight logging statements (stdout prints) that do NOT change returned values.
- Add minimal comments.

You may NOT:
- Add new files.
- Move or rename files.
- Rename functions or variables used by other modules.

---

## 2) Do NOT Touch These Files
Do NOT modify:
- `benchmark.py`
- `submit.py`
- Any evaluation/scoring logic
- Any config files required by the repo

No exceptions.

---

## 3) Signature & Output Contracts (Non-Negotiable)
- DO NOT change the function signature of `generate_hybrid()` in ANY way.
- DO NOT change the return type or shape of what `generate_hybrid()` returns.
- DO NOT change tool schemas, tool names, or expected tool-call formats.
- DO NOT change any public interfaces used by `benchmark.py` or `submit.py`.

---

## 4) Dependencies / Imports
- Prefer using existing imports.
- If a new import is absolutely necessary, it MUST be from the Python standard library only.
- Do NOT add new third-party dependencies.
- Do NOT require new environment variables.
- Do NOT hardcode API keys or secrets.

---

## 5) Security / Secrets
- Never write secrets into code.
- Never print environment variables.
- Never introduce code that would log keys/tokens.

---

## 6) Routing Improvements (What to Implement)
Improve hybrid routing by making minimal, safe edits such as:

- Complexity detection (e.g., multi-step instructions, conditionals, long prompts, log-like text)
- Dynamic confidence thresholds
- One local retry with stricter formatting if tool-call output is invalid
- Fallback to cloud only when local is likely to fail
- Prefer local execution whenever safe

Avoid over-engineering. Keep changes small and testable.

---

## 7) Logging Rules
Logging is allowed ONLY if:
- It does not alter outputs
- It is concise and useful
- It can be easily disabled (e.g., a `DEBUG = False` flag in `main.py`)

Do not add noisy logs that break readability.

---

## 8) Change Management
For every edit:
- Keep diffs minimal.
- Do not refactor unrelated code.
- Do not reformat entire files.
- Avoid broad “cleanup” changes.

---

## 9) Success Criteria
After changes:
- `python benchmark.py` must run successfully without modifying `benchmark.py`.
- `python submit.py --team "X" --location "Y"` must run successfully without modifying `submit.py`.

---

## 10) If Unsure
If you are unsure about any contract/return format:
- Do NOT guess.
- Inspect the existing code usage (how `benchmark.py` calls it) without editing those files.
- Make the smallest safe change possible.

---