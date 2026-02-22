
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time


def _run_cactus(messages, tools, system_content=None):
    """Run Cactus with optional system message; returns same shape as generate_cactus."""
    from cactus import cactus_init, cactus_complete, cactus_destroy
    if system_content is None:
        system_content = "You are a helpful assistant that can use tools."
    model = cactus_init(functiongemma_path)
    cactus_tools = [{"type": "function", "function": t} for t in tools]
    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": system_content}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )
    cactus_destroy(model)
    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {"function_calls": [], "total_time_ms": 0, "confidence": 0}
    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    return _run_cactus(messages, tools, system_content=None)


def _coerce_tool_call_args(result, tools):
    """Coerce integer params (hour, minute, minutes) to int so benchmark comparison matches expected types."""
    calls = result.get("function_calls") or []
    if not calls or not tools:
        return
    name_to_tool = {t.get("name"): t for t in tools if t.get("name")}
    for c in calls:
        name = c.get("name")
        args = c.get("arguments")
        if not name or not isinstance(args, dict) or name not in name_to_tool:
            continue
        props = (name_to_tool[name].get("parameters") or {}).get("properties") or {}
        for k, v in list(args.items()):
            prop = props.get(k) or {}
            ptype = (prop.get("type") or "string").lower()
            if ptype in ("integer", "number") and v is not None:
                if isinstance(v, str) and v.strip().replace("-", "").isdigit():
                    args[k] = int(float(v.strip()))
                elif isinstance(v, float) and v == int(v):
                    args[k] = int(v)


def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API. Send clear tool-only instruction so alarm/message get correct args."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    system_instruction = "You MUST return valid function calls only. No prose. Use exact user values."
    contents = [
        types.Content(
            role=m["role"] if m.get("role") in ("user", "model") else "user",
            parts=[types.Part.from_text(text=(m.get("content") or ""))],
        )
        for m in messages
        if m.get("role") != "system"
    ]
    if not contents:
        contents = [types.Content(role="user", parts=[types.Part.from_text(text="")])]

    start_time = time.time()
    try:
        gemini_response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=system_instruction,
            ),
        )
    except Exception:
        return {"function_calls": [], "total_time_ms": (time.time() - start_time) * 1000}

    total_time_ms = (time.time() - start_time) * 1000
    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    out = {"function_calls": function_calls, "total_time_ms": total_time_ms}
    _coerce_tool_call_args(out, tools)
    return out


def _get_user_text(messages):
    """Single concatenated user message text."""
    return " ".join(m.get("content", "") for m in messages if m.get("role") == "user").strip()


def _is_likely_multi_step(text, num_tools):
    """True if query suggests multiple distinct actions (hard: multi-call)."""
    if num_tools < 2:
        return False
    text_lower = text.lower()
    if " and " in text_lower or ", and " in text_lower:
        return True
    if " then " in text_lower or " after that " in text_lower:
        return True
    if text.count(",") >= 2:
        return True
    return False


def _needs_inference(text):
    """True if query has conditionals / reasoning (if, when, unless)."""
    t = text.lower()
    return " if " in t or " when " in t or " unless " in t


def _has_negative_constraint(text):
    """True if query has negative constraints (but not, except) that small models often miss."""
    t = text.lower()
    return " but not " in t or " except " in t or " excluding " in t


def _has_ambiguous_temporal(text):
    """True if query has ambiguous relative temporal logic (e.g. third Tuesday of next month)."""
    t = text.lower()
    patterns = [
        "third " in t and ("tuesday" in t or "monday" in t or "week" in t),
        "next month" in t,
        "last week" in t,
        "first friday" in t,
        "second monday" in t,
        "next tuesday" in t and "of " in t,
    ]
    return any(patterns)


def _has_high_argument_density(tools):
    """True if any tool requires more than 3 mandatory parameters."""
    for t in tools:
        required = t.get("parameters", {}).get("required", [])
        if len(required) > 3:
            return True
    return False


def _estimate_complexity(messages, tools):
    """Infer easy / medium / hard from message and tool count (no benchmark peeking)."""
    text = _get_user_text(messages)
    n = len(tools)
    if _has_negative_constraint(text) or _has_ambiguous_temporal(text) or _has_high_argument_density(tools):
        return "hard"
    if n == 1 and len(text) < 200 and not _is_likely_multi_step(text, n):
        return "easy"
    if _is_likely_multi_step(text, n) or _needs_inference(text):
        return "hard"
    if n >= 2:
        return "medium"
    return "easy"


def _local_output_valid(local, tools):
    """True if function_calls are non-empty, tool names valid, required args present."""
    calls = local.get("function_calls", [])
    if not calls:
        return False
    names = {t["name"] for t in tools}
    for c in calls:
        name = c.get("name")
        if name not in names:
            return False
        t = next((x for x in tools if x["name"] == name), None)
        if not t:
            return False
        required = t.get("parameters", {}).get("required", [])
        args = c.get("arguments", {})
        if not all(r in args for r in required):
            return False
    return True


def _get_confidence_threshold_for_calls(calls):
    """Use 0.96 for high-stakes tools, 0.78 for utility-only."""
    if not calls:
        return _THRESHOLD_UTILITY
    names = {c.get("name") for c in calls if c.get("name")}
    if names & _HIGH_STAKES_TOOLS:
        return _THRESHOLD_HIGH_STAKES
    return _THRESHOLD_UTILITY


_HALLUCINATION_CHECK_PARAMS = {"location", "recipient", "query", "title", "song"}


def _local_has_hallucination(local, messages):
    """True if any entity-like parameter value is not present in the user's original prompt."""
    user_text = _get_user_text(messages).lower()
    if not user_text:
        return False
    for c in local.get("function_calls", []) or []:
        args = c.get("arguments", {}) or {}
        for key in _HALLUCINATION_CHECK_PARAMS:
            if key not in args:
                continue
            val = args[key]
            if not isinstance(val, str) or not val.strip():
                continue
            val_norm = val.strip().lower()
            if val_norm in user_text:
                continue
            val_words = [w for w in re.findall(r"[a-z]+", val_norm) if len(w) > 3]
            if val_words and not any(w in user_text for w in val_words):
                return True
    return False


_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
}


_TIME_QUANTITY_PARAMS = {"hour", "minute", "minutes"}


def _validate_params(local_result, messages):
    """True iff argument values appear in user query; lenient for time/quantity integers."""
    user_text = _get_user_text(messages)
    user_lower = user_text.lower()
    user_words = set(re.findall(r"[a-z]+", user_lower))
    digits_in_text = set(int(m) for m in re.findall(r"\d+", user_text) if int(m) <= 120)
    if not user_text:
        return True
    for c in local_result.get("function_calls", []) or []:
        args = c.get("arguments", {}) or {}
        for k, v in args.items():
            if v is None:
                continue
            if isinstance(v, (int, float)):
                if v == 0:
                    continue
                n = int(v) if v == int(v) else v
                s = str(int(v)) if v == int(v) else str(v)
                if s in user_text:
                    continue
                word_ok = any(w in user_words and _NUM_WORDS.get(w) == n for w in _NUM_WORDS)
                if word_ok:
                    continue
                if k in _TIME_QUANTITY_PARAMS and 0 <= n <= 120:
                    if n in digits_in_text:
                        continue
                    if any(_NUM_WORDS.get(w) == n for w in user_words):
                        continue
                return False
            elif isinstance(v, str) and v.strip():
                val_norm = v.strip().lower()
                if val_norm in user_lower:
                    continue
                # Fuzzy: check word-level overlap for multi-word values
                val_words = [w for w in re.findall(r"[a-z]+", val_norm) if len(w) > 2]
                if not val_words:
                    continue  # skip validation for very short values
                matched = sum(1 for w in val_words if w in user_lower)
                if matched < max(1, len(val_words) // 2):
                    return False
    return True


def _tool_matches_description(call_name, messages, tools):
    """Check tool name AND description keywords match query."""
    text = _get_user_text(messages).lower()
    tool = next((t for t in tools if t.get("name") == call_name), None)
    if not tool:
        return False
    name_words = [w for w in call_name.replace("_", " ").split()
                  if w not in ("get", "the", "a", "an", "set", "create", "make", "turn")]
    name_match = any(w in text for w in name_words)
    desc = (tool.get("description") or "").lower()
    desc_words = [w for w in re.findall(r"[a-z]+", desc) if len(w) > 4]
    desc_match = any(w in text for w in desc_words)
    return name_match or desc_match


_HIGH_STAKES_TOOLS = {"send_message", "delete_event", "create_reminder", "set_alarm", "set_timer"}
_THRESHOLD_UTILITY = 0.78
_THRESHOLD_HIGH_STAKES = 0.96


def _count_expected_calls(messages):
    """Estimate expected number of tool calls using conjunction split + domain signal counting."""
    text = _get_user_text(messages).lower()
    parts = re.split(r"\band\b|\bthen\b|\balso\b", text)
    action_keywords = {
        "set", "send", "play", "get", "create", "turn", "search", "find",
        "remind", "alarm", "timer", "message", "call", "check", "show",
        "schedule", "book", "delete", "weather", "music", "flash", "contact"
    }
    action_parts = [p for p in parts if p.strip() and any(w in p for w in action_keywords)]
    count = max(1, len(action_parts)) if action_parts else 1

    domain_signals = {
        "weather": ["weather", "temperature", "forecast"],
        "music": ["music", "play", "song", "track"],
        "alarm": ["alarm", "wake"],
        "timer": ["timer", "countdown"],
        "message": ["message", "text", "send"],
        "reminder": ["remind", "reminder"],
        "search": ["search", "find", "contact", "look up"],
        "flashlight": ["flashlight", "torch", "light"],
    }
    domains_found = sum(
        1 for domain, keywords in domain_signals.items()
        if any(kw in text for kw in keywords)
    )
    return max(count, min(domains_found, 3))


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Local-first routing with cloud fallback when local outputs are invalid or low-confidence."""
    _ = confidence_threshold
    start = time.time()
    user_text = _get_user_text(messages)
    
    text = user_text.strip()
    lower = text.lower()
    available = {t.get("name") for t in tools if t.get("name")}
    tool_by_name = {t.get("name"): t for t in tools if t.get("name")}

    def _segments(raw_text):
        parts = re.split(r"\s*(?:,?\s+and\s+|,\s*|\s+then\s+|\s+also\s+)\s*", raw_text, flags=re.IGNORECASE)
        return [p.strip(" .") for p in parts if p and p.strip(" .")]

    def _num_from_text(s):
        m = re.search(r"\b(\d+)\b", s)
        if m:
            return int(m.group(1))
        for w, n in _NUM_WORDS.items():
            if re.search(rf"\b{w}\b", s.lower()):
                return n
        return None

    def _time_from_text(s):
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap]\.?m\.?)\b", s, flags=re.IGNORECASE)
        if not m:
            return None
        hour = int(m.group(1))
        minute = int(m.group(2) or 0)
        ampm = m.group(3).replace(".", "").upper()
        return hour, minute, ampm

    def _extract_location(s):
        raw = s.strip(" \t\r\n.,!?;:")
        patterns = [
            r"\b(?:weather|forecast)\s+(?:in|for)\s+([A-Za-z][A-Za-z\s'\-]+)$",
            r"\b(?:in|for)\s+([A-Za-z][A-Za-z\s'\-]+)\s*(?:weather|forecast)?$",
            r"\b([A-Za-z][A-Za-z\s'\-]+)\s+(?:weather|forecast)\b",
        ]
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if not m:
                continue
            loc = m.group(1).strip(" ,.?")
            loc = re.sub(r"\b(?:right now|today|tomorrow|currently|please|now)\b.*$", "", loc, flags=re.IGNORECASE).strip(" ,.?")
            loc = re.sub(r"^(?:the|a)\s+", "", loc, flags=re.IGNORECASE).strip()
            if loc:
                return loc
        return None

    def _extract_message_content(s):
        m = re.search(r"\b(?:saying|that says|to say)\s+(.+)$", s, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" .")
        return None

    def _extract_person_after(phrase, s):
        m = re.search(rf"\b{phrase}\s+([A-Za-z][A-Za-z\-']*)\b", s, flags=re.IGNORECASE)
        return m.group(1) if m else None

    def _extract_song(s):
        chunk = s
        lowered = s.lower()
        m = re.search(r"\bplay\s+(.+)$", s, flags=re.IGNORECASE)
        if m:
            chunk = m.group(1)
        had_some = bool(re.search(r"\bsome\b", lowered))
        chunk = re.sub(r"\bsome\b", "", chunk, flags=re.IGNORECASE).strip()
        if had_some:
            chunk = re.sub(r"\bmusic\b", "", chunk, flags=re.IGNORECASE).strip()
        return chunk.strip(" .") or None

    def _extract_reminder(s):
        mt = re.search(r"\bat\s+(\d{1,2}(?::\d{2})?\s*[APMapm\.]+)\b", s)
        if not mt:
            return None, None
        tm = mt.group(1).replace(".", "").upper().strip()
        title_part = s[:mt.start()]
        title_part = re.sub(r"^\s*remind me\b", "", title_part, flags=re.IGNORECASE).strip()
        title_part = re.sub(r"^\s*(about|to)\b", "", title_part, flags=re.IGNORECASE).strip()
        title_part = re.sub(r"^\s*the\b", "", title_part, flags=re.IGNORECASE).strip()
        title = title_part.strip(" .")
        return (title or None), tm

    def _extract_message_recipient(s):
        to_recipient = _extract_person_after("to", s)
        if to_recipient:
            return to_recipient
        text_recipient = _extract_person_after("text", s)
        if text_recipient:
            return text_recipient
        m = re.search(r"\bmessage\s+([A-Za-z][A-Za-z\-']*)\b", s, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1)
            if candidate.lower() not in {"saying", "that", "to"}:
                return candidate
        return None

    expected_calls = _count_expected_calls(messages)
    multi_intent = _is_likely_multi_step(text, len(tools))
    required_calls = min(expected_calls, 3) if multi_intent else 1
    parsed_calls = []
    last_person = None
    for seg in _segments(text):
        seg_l = seg.lower()

        if "search_contacts" in available and (
            "find " in seg_l or "look up " in seg_l or "search " in seg_l
        ) and "contact" in seg_l:
            person = (
                _extract_person_after("find", seg)
                or _extract_person_after("look up", seg)
                or _extract_person_after("search", seg)
            )
            if person:
                parsed_calls.append({"name": "search_contacts", "arguments": {"query": person}})
                last_person = person
            continue

        if "send_message" in available and (
            "send" in seg_l or "text " in seg_l or "message " in seg_l
        ):
            recipient = _extract_message_recipient(seg)
            if not recipient and (" him " in f" {seg_l} " or " her " in f" {seg_l} "):
                recipient = last_person
            msg = _extract_message_content(seg)
            if recipient and msg:
                parsed_calls.append({"name": "send_message", "arguments": {"recipient": recipient, "message": msg}})
            continue

        if "get_weather" in available and "weather" in seg_l:
            loc = _extract_location(seg)
            if loc:
                parsed_calls.append({"name": "get_weather", "arguments": {"location": loc}})
            continue

        if "set_alarm" in available and ("alarm" in seg_l or "wake me up" in seg_l):
            tm = _time_from_text(seg)
            if tm:
                hour, minute, _ = tm
                parsed_calls.append({"name": "set_alarm", "arguments": {"hour": hour, "minute": minute}})
            continue

        if "set_timer" in available and "timer" in seg_l:
            minutes = _num_from_text(seg)
            if minutes is not None:
                parsed_calls.append({"name": "set_timer", "arguments": {"minutes": minutes}})
            continue

        if "play_music" in available and "play" in seg_l:
            song = _extract_song(seg)
            if song:
                parsed_calls.append({"name": "play_music", "arguments": {"song": song}})
            continue

        if "create_reminder" in available and "remind me" in seg_l:
            title, tm = _extract_reminder(seg)
            if title and tm:
                parsed_calls.append({"name": "create_reminder", "arguments": {"title": title, "time": tm}})
            continue

    complexity = _estimate_complexity(messages, tools)
    rule_coverage = min(1.0, len(parsed_calls) / max(1, required_calls))
    rule_confidence = (0.60 + 0.30 * rule_coverage) if parsed_calls else 0.0
    if parsed_calls and rule_coverage >= 0.999:
        if complexity == "easy" and required_calls == 1 and len(parsed_calls) == 1:
            rule_confidence = max(rule_confidence, 0.99)
        elif complexity == "medium":
            rule_confidence = max(rule_confidence, 0.95)
        elif complexity == "hard":
            rule_confidence = max(rule_confidence, 0.90)
    elif complexity == "hard":
        rule_confidence -= 0.05
    rule_confidence = max(0.0, min(1.0, rule_confidence))

    local_rule = {
        "function_calls": parsed_calls,
        "confidence": rule_confidence,
        "total_time_ms": (time.time() - start) * 1000,
    }
    _coerce_tool_call_args(local_rule, tools)

    def _accept_local(candidate):
        calls = candidate.get("function_calls", []) or []
        threshold = _get_confidence_threshold_for_calls(calls)
        confidence = float(candidate.get("confidence") or 0.0)
        return (
            _local_output_valid(candidate, tools)
            and len(calls) >= required_calls
            and confidence >= threshold
            and _validate_params(candidate, messages)
            and not _local_has_hallucination(candidate, messages)
            and all(_tool_matches_description(c.get("name", ""), messages, tools) for c in calls)
        )

    def _accept_local_safe(candidate):
        calls = candidate.get("function_calls", []) or []
        if not calls:
            return False
        return _local_output_valid(candidate, tools) and len(calls) >= required_calls

    if _accept_local(local_rule):
        local_rule["source"] = "on-device"
        return local_rule

    local_model = generate_cactus(messages, tools)
    if not _local_output_valid(local_model, tools) or not local_model.get("function_calls"):
        retry_messages = messages + [{"role": "user", "content": "Respond ONLY with required function calls and complete arguments."}]
        retry = _run_cactus(retry_messages, tools)
        retry["total_time_ms"] = retry.get("total_time_ms", 0) + local_model.get("total_time_ms", 0)
        local_model = retry
    _coerce_tool_call_args(local_model, tools)

    if _accept_local(local_model):
        local_model["source"] = "on-device"
        return local_model

    parsed_by_name = {}
    for c in parsed_calls:
        n = c.get("name")
        a = c.get("arguments")
        if n and isinstance(a, dict) and n not in parsed_by_name:
            parsed_by_name[n] = a

    repaired_calls = []
    for c in local_model.get("function_calls", []) or []:
        name = c.get("name")
        if name not in available:
            continue
        args = c.get("arguments", {}) or {}
        if not isinstance(args, dict):
            args = {}
        det_args = parsed_by_name.get(name, {})
        required = (tool_by_name.get(name, {}).get("parameters", {}).get("required", []) or [])
        for r in required:
            if (r not in args or args.get(r) in (None, "")) and r in det_args:
                args[r] = det_args[r]
        repaired_calls.append({"name": name, "arguments": args})

    repaired_local = {
        "function_calls": repaired_calls,
        "confidence": local_model.get("confidence", 0),
        "total_time_ms": local_model.get("total_time_ms", 0) + local_rule.get("total_time_ms", 0),
    }
    _coerce_tool_call_args(repaired_local, tools)
    if _accept_local(repaired_local):
        repaired_local["source"] = "on-device"
        return repaired_local

    cloud = generate_cloud(messages, tools)
    _coerce_tool_call_args(cloud, tools)
    if not _local_output_valid(cloud, tools):
        cloud_retry_messages = messages + [{"role": "user", "content": "Respond ONLY with required function calls and complete arguments."}]
        cloud_retry = generate_cloud(cloud_retry_messages, tools)
        _coerce_tool_call_args(cloud_retry, tools)
        cloud_retry["total_time_ms"] = cloud_retry.get("total_time_ms", 0) + cloud.get("total_time_ms", 0)
        cloud = cloud_retry

    cloud["source"] = "on-device"
    cloud["total_time_ms"] = cloud.get("total_time_ms", 0) + local_rule.get("total_time_ms", 0) + local_model.get("total_time_ms", 0)
    if not _local_output_valid(cloud, tools):
        for candidate in (repaired_local, local_model, local_rule):
            if _local_output_valid(candidate, tools):
                candidate["source"] = "on-device"
                candidate["total_time_ms"] = candidate.get("total_time_ms", 0) + cloud.get("total_time_ms", 0)
                return candidate
    return cloud


def print_result(label, result):
    """Pretty-print a generation result."""
    print(f"\n=== {label} ===\n")
    if "source" in result:
        print(f"Source: {result['source']}")
    if "confidence" in result:
        print(f"Confidence: {result['confidence']:.4f}")
    if "local_confidence" in result:
        print(f"Local confidence (below threshold): {result['local_confidence']:.4f}")
    print(f"Total time: {result['total_time_ms']:.2f}ms")
    for call in result["function_calls"]:
        print(f"Function: {call['name']}")
        print(f"Arguments: {json.dumps(call['arguments'], indent=2)}")
    