import math
import re
import unicodedata

_INVISIBLE = re.compile("[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff]")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_UPI = re.compile(r"\b[\w.\-]{2,}@(?:ybl|ibl|axl|paytm|okaxis|okhdfcbank|oksbi|okicici|upi)\b", re.I)
_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{4}[ -]?\d{5}(?!\d)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)

_INJECTION = [
    ("override", re.compile(r"\b(ignore|disregard|forget|bypass|override|skip)\b.{0,40}\b(previous|prior|above|earlier|all|any|the|your)\b.{0,40}\b(instruction|rule|guardrail|polic|limit|constraint|prompt|safety)", re.I | re.S)),
    ("prompt_probe", re.compile(r"\b(system|developer|hidden|initial)\s+(prompt|message|instruction)s?\b|\breveal\b.{0,30}\b(prompt|instruction|key|secret|password)", re.I | re.S)),
    ("role_change", re.compile(r"\b(you are now|act as|pretend to be|developer mode|jailbreak|do anything now)\b", re.I)),
    ("tool_spoof", re.compile(r"</?\s*(system|assistant|tool|function)[^>]*>|\b(function_call|tool_call|function_response)\b", re.I)),
    ("exec_request", re.compile(r"\b(execute|run|eval)\b.{0,30}\b(sql|shell|command|script|code|query)\b|\b(drop|delete)\s+table\b", re.I | re.S)),
    ("secret_request", re.compile(r"\b(api[_ -]?key|access[_ -]?code|session[_ -]?secret|password|token)\b", re.I)),
]

POLICY_NUMBERS = {2, 3, 5, 7, 14, 25, 50, 80, 2000, 75000}


def clean_text(text: object, limit: int = 300) -> str:
    value = unicodedata.normalize("NFKC", str(text))
    value = _INVISIBLE.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()[:limit]


def _luhn(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        number = int(char)
        if index % 2:
            number = number * 2 - 9 if number > 4 else number * 2
        total += number
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    kinds: list[str] = []

    def replace(pattern: re.Pattern, label: str, check=None):
        nonlocal text

        def swap(match: re.Match) -> str:
            if check and not check(match.group(0)):
                return match.group(0)
            kinds.append(label)
            return f"[{label} removed]"

        text = pattern.sub(swap, text)

    replace(_EMAIL, "email")
    replace(_UPI, "payment id")
    replace(_CARD, "card number", lambda value: 13 <= len(re.sub(r"\D", "", value)) <= 19 and _luhn(re.sub(r"\D", "", value)))
    replace(_AADHAAR, "id number")
    replace(_PAN, "tax id")
    replace(_PHONE, "phone number")
    return text, kinds


def injection_flags(text: str) -> list[str]:
    return [label for label, pattern in _INJECTION if pattern.search(text)]


def sanitize_untrusted(text: object, limit: int = 300) -> tuple[str, list[str], list[str]]:
    clean, redactions = redact_pii(clean_text(text, limit))
    return clean, redactions, injection_flags(clean)


def safe_catalog_text(text: object, limit: int = 200) -> str:
    clean, _, flags = sanitize_untrusted(text, limit)
    return "Unavailable product text" if flags or not clean else clean


def strip_links(text: str) -> str:
    return _URL.sub("[link removed]", text)


_NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")
_IDENTIFIER = re.compile(r"\b[A-Z]{2,4}-[A-Z0-9]{2,}(?:-[A-Z0-9]+)*\b")


def numbers_in(text: str) -> list[float]:
    values = []
    for match in _NUMBER.finditer(_IDENTIFIER.sub(" ", text)):
        try:
            values.append(float(match.group(1).replace(",", "") + (match.group(2) or "")))
        except ValueError:
            continue
    return values


def _collect(value, out: set[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            out.add(float(value))
            if 0 <= value <= 1:
                out.add(round(float(value) * 100, 1))
    elif isinstance(value, dict):
        for item in value.values():
            _collect(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect(item, out)


def allowed_numbers(*sources, extra=()) -> set[float]:
    out: set[float] = set(POLICY_NUMBERS) | {float(item) for item in extra}
    for source in sources:
        _collect(source, out)
        if isinstance(source, dict) and source and all(isinstance(item, dict) for item in source.values()):
            columns: dict[str, list[float]] = {}
            for item in source.values():
                for key, number in item.items():
                    if isinstance(number, (int, float)) and not isinstance(number, bool) and math.isfinite(number):
                        columns.setdefault(key, []).append(float(number))
            for numbers in columns.values():
                out.add(round(sum(numbers), 1))
                out.add(round(sum(numbers) / len(numbers), 1))
                out.add(float(len(numbers)))
    return out


def ungrounded_numbers(text: str, allowed: set[float]) -> list[float]:
    missing = []
    for number in numbers_in(text):
        if not any(abs(number - known) <= max(0.51, abs(known) * 0.01) for known in allowed):
            missing.append(number)
    return missing


def screen_model_text(text: object, allowed: set[float], limit: int = 600) -> tuple[str | None, list[str]]:
    """Return safe text, or None plus the reasons it was not trusted."""
    if not isinstance(text, str):
        return None, ["not_text"]
    clean, redactions = redact_pii(strip_links(clean_text(text, limit * 2)))
    reasons = []
    if len(clean) > limit:
        reasons.append("too_long")
    if injection_flags(clean):
        reasons.append("unsafe_instruction")
    if redactions:
        reasons.append("personal_data")
    unknown = ungrounded_numbers(clean, allowed)
    if unknown:
        reasons.append("numbers_not_in_evidence")
    return (None if reasons else clean), reasons
