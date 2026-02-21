
import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, re, time
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


def _run_cactus(messages, tools, system_content=None):
    """Run Cactus with optional system message; returns same shape as generate_cactus."""
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

    user_contents = [m["content"] for m in messages if m.get("role") == "user"]
    tool_instruction = "Tool calls only. No prose. Use user's values for args."
    contents = [tool_instruction + "\n\n" + (user_contents[0] or "")] if user_contents else [tool_instruction]
    if len(user_contents) > 1:
        contents.extend(user_contents[1:])

    start_time = time.time()
    try:
        gemini_response = client.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(tools=gemini_tools),
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


def _is_reminder_with_time(messages, tools):
    """Reminder+time queries often fail on-device; route to cloud for F1."""
    names = {t.get("name") for t in tools}
    if "create_reminder" not in names:
        return False
    text = _get_user_text(messages).upper()
    if "AM" not in text and "PM" not in text:
        return False
    return ":" in _get_user_text(messages) or " AT " in text


def _should_fallback_medium(messages, tools):
    """True only for reminder among 4 tools with time in query."""
    names = {t.get("name") for t in tools if t.get("name")}
    if "create_reminder" not in names or len(tools) != 4:
        return False
    text = _get_user_text(messages).upper()
    has_time = "AM" in text or "PM" in text or ":" in text
    return has_time


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


# Params whose values should appear in user prompt (entity-like: names, places).
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
            if val_norm not in user_text:
                return True
    return False


# Word form of numbers for phonetic equivalence in _validate_params (e.g. "five" vs 5).
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
                if val_norm not in user_lower:
                    return False
    return True


def _tool_name_matches_query(call_name, messages):
    """Check that the called tool name has at least one meaningful word in the query."""
    text = _get_user_text(messages).lower()
    words = [w for w in call_name.replace("_", " ").split()
             if w not in ("get", "the", "a", "an", "set", "create", "make")]
    return any(w in text for w in words) if words else True


_UTILITY_TOOLS = {"get_weather", "turn_on_flashlight", "play_music", "search_contacts"}
_HIGH_STAKES_TOOLS = {"send_message", "delete_event", "create_reminder", "set_alarm", "set_timer"}
_THRESHOLD_UTILITY = 0.78
_THRESHOLD_HIGH_STAKES = 0.96


def _count_expected_calls(messages):
    """Estimate expected number of tool calls from query conjunction count."""
    text = _get_user_text(messages).lower()
    parts = re.split(r"\band\b|\bthen\b|\balso\b", text)
    return max(1, len([p for p in parts if p.strip()]))


def generate_hybrid(messages, tools, confidence_threshold=0.99):
    """Always try on-device first; fall back to cloud only when local is invalid or low confidence."""
    complexity = _estimate_complexity(messages, tools)

    if complexity == "easy" and _is_reminder_with_time(messages, tools):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud
    if complexity == "medium" and _should_fallback_medium(messages, tools):
        cloud = generate_cloud(messages, tools)
        cloud["source"] = "cloud (fallback)"
        return cloud

    local = generate_cactus(messages, tools)
    if not _local_output_valid(local, tools) or not (local.get("function_calls")):
        first_time_ms = local.get("total_time_ms", 0)
        skip_retry = (
            (complexity == "hard" and local.get("confidence", 0) < 0.88)
            or (complexity == "medium" and first_time_ms > 300)
        )
        if not skip_retry:
            retry_content = "System: Format your response ONLY as a tool call JSON. No prose. Request: " + _get_user_text(messages)
            retry_messages = [{"role": "user", "content": retry_content}]
            local = _run_cactus(retry_messages, tools)
            local["total_time_ms"] = local.get("total_time_ms", 0) + first_time_ms

    if _local_output_valid(local, tools) and (_local_has_hallucination(local, messages) or not _validate_params(local, messages)):
        local = {"function_calls": [], "confidence": 0, "total_time_ms": local.get("total_time_ms", 0)}

    user_len = len(_get_user_text(messages))
    calls = local.get("function_calls", [])
    call_names = {c.get("name") for c in calls if c.get("name")}
    if complexity == "hard" and len(tools) == 2:
        threshold = 0.75
    elif complexity == "medium" and call_names and call_names <= _UTILITY_TOOLS:
        threshold = 0.65
    elif complexity == "medium" and user_len < 60:
        threshold = 0.75
    else:
        threshold = _get_confidence_threshold_for_calls(calls)

    if local["confidence"] >= threshold and _local_output_valid(local, tools):
        all_calls_match = all(
            _tool_name_matches_query(c.get("name", ""), messages)
            for c in local.get("function_calls", [])
        )
        enough_calls = True
        if complexity == "hard":
            enough_calls = len(local.get("function_calls", [])) >= _count_expected_calls(messages)
        if all_calls_match and enough_calls:
            local["source"] = "on-device"
            _coerce_tool_call_args(local, tools)
            return local

    cloud = generate_cloud(messages, tools)
    if complexity == "hard":
        expected = _count_expected_calls(messages)
        actual = len(cloud.get("function_calls", []))
        if actual < expected:
            retry_msgs = [{
                "role": "user",
                "content": f"Make exactly {expected} tool calls for this request: " + _get_user_text(messages),
            }]
            cloud2 = generate_cloud(retry_msgs, tools)
            if len(cloud2.get("function_calls", [])) > actual:
                cloud2["source"] = "cloud (fallback)"
                cloud2["local_confidence"] = local.get("confidence", 0)
                cloud2["total_time_ms"] = cloud2.get("total_time_ms", 0) + local.get("total_time_ms", 0)
                return cloud2
    cloud["source"] = "cloud (fallback)"
    cloud["local_confidence"] = local.get("confidence", 0)
    cloud["total_time_ms"] = cloud.get("total_time_ms", 0) + local.get("total_time_ms", 0)
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


############## Example usage ##############

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["location"],
        },
    }]

    messages = [
        {"role": "user", "content": "What is the weather in San Francisco?"}
    ]

    on_device = generate_cactus(messages, tools)
    print_result("FunctionGemma (On-Device Cactus)", on_device)

    cloud = generate_cloud(messages, tools)
    print_result("Gemini (Cloud)", cloud)

    hybrid = generate_hybrid(messages, tools)
    print_result("Hybrid (On-Device + Cloud Fallback)", hybrid)
    