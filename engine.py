"""
Dhan Saathi - core logic (no UI).
Retrieval over the sample knowledge base, crop-stage calculation,
weather fetch + rule-based advisories, and input guardrails.
Everything here is deterministic, so it can be tested without an API key.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import requests

DATA = Path(__file__).parent / "data"

# ---------------------------------------------------------------- knowledge base
KB: list[dict] = json.loads((DATA / "knowledge_base.json").read_text(encoding="utf-8"))

# Hindi / Punjabi farmer words mapped to English KB vocabulary so that
# a question typed in Hinglish or Devanagari still retrieves the right entry.
SYNONYMS = {
    "khad": "fertiliser", "khaad": "fertiliser", "खाद": "fertiliser", "यूरिया": "urea",
    "pani": "irrigation", "paani": "irrigation", "पानी": "irrigation", "सिंचाई": "irrigation",
    "keeda": "pest", "keede": "pest", "kida": "pest", "कीड़ा": "pest", "कीड़े": "pest", "ਕੀੜੇ": "pest",
    "rog": "disease", "bimari": "disease", "बीमारी": "disease", "रोग": "disease", "ਬਿਮਾਰੀ": "disease",
    "beej": "seed", "बीज": "seed", "ਬੀਜ": "seed",
    "ropai": "transplanting", "रोपाई": "transplanting", "ਲੁਆਈ": "transplanting",
    "kataai": "harvest", "katai": "harvest", "कटाई": "harvest", "ਵਾਢੀ": "harvest",
    "parali": "stubble", "पराली": "stubble", "ਪਰਾਲੀ": "stubble",
    "baarish": "rain", "barish": "rain", "बारिश": "rain", "ਮੀਂਹ": "rain",
    "spray": "spray", "chhidkav": "spray", "छिड़काव": "spray",
    "jhulsa": "blast", "khaira": "khaira", "खैरा": "khaira", "tela": "planthopper",
    "zehar": "poison", "ज़हर": "poison", "जहर": "poison",
    "safed": "white", "सफेद": "white", "peela": "yellow", "पीला": "yellow", "पीली": "yellow",
    "yuriya": "urea", "uria": "urea", "daalna": "apply", "dalna": "apply", "dalein": "apply",
    "sookh": "dries", "sukh": "dries", "sookha": "dries", "beech": "central", "beech-wala": "central",
    "patta": "leaf", "patte": "leaves", "pattiyan": "leaves", "पत्ते": "leaves", "पत्तियाँ": "leaves", "धब्बे": "spots",
    "dhabbe": "spots", "bhoore": "brown", "भूरे": "brown",
    "leaves": "leaf", "dhan": "rice", "धान": "rice", "ਝੋਨਾ": "rice", "chawal": "rice",
}
STOP = set("""a an the is are was were be to of in on for at by with and or my me i we our
what which when how why do does can should will it this that these those there please tell
about give from have has had any some your you about field crop rice paddy""".split())


def _stem(w: str) -> str:
    """Very light English stemmer so 'shoots'/'shoot' and 'drying'/'dries'/'dried' match.
    (Edge case found in live testing: 'central shoots are drying' retrieved BPH instead of stem borer.)"""
    if not w.isascii() or len(w) <= 4:
        return w
    for suf, rep in (("ies", "y"), ("ied", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)] + rep
    return w


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[\wऀ-ॿ਀-੿]+", text.lower())
    out = []
    for w in words:
        w = _stem(SYNONYMS.get(w, w))
        if w not in STOP and len(w) > 1:
            out.append(w)
    return out


_DOCS = [_tokens(f"{d['title']} {d['keywords']} {d['keywords']} {d['text']}") for d in KB]
_DF = Counter(t for doc in _DOCS for t in set(doc))
_AVGDL = sum(len(d) for d in _DOCS) / len(_DOCS)


def retrieve(query: str, k: int = 3, stage: str | None = None) -> list[tuple[dict, float]]:
    """BM25 retrieval over the knowledge base, with a small boost for the current crop stage."""
    q = _tokens(query)
    n = len(_DOCS)
    scored = []
    for doc, entry in zip(_DOCS, KB):
        tf = Counter(doc)
        s = 0.0
        for t in q:
            if t not in tf:
                continue
            idf = math.log(1 + (n - _DF[t] + 0.5) / (_DF[t] + 0.5))
            s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(doc) / _AVGDL))
        if s > 0 and stage and entry["stage"] == stage:
            s *= 1.15
        scored.append((entry, round(s, 2)))
    scored.sort(key=lambda x: -x[1])
    return [x for x in scored[:k] if x[1] > 0]


# ---------------------------------------------------------------- crop stage
STAGES = [
    ("nursery", "Nursery", "नर्सरी"),
    ("transplanting", "Transplanting / establishment", "रोपाई"),
    ("tillering", "Tillering", "कल्ले निकलना"),
    ("panicle initiation", "Panicle initiation / booting", "बाली बनना"),
    ("flowering", "Flowering", "फूल आना"),
    ("grain filling", "Grain filling (milky-dough)", "दाना भरना"),
    ("maturity", "Maturity / harvest", "पकना / कटाई"),
]


def stage_from_dat(dat: int) -> str:
    """Approximate stage of a ~125-day variety from days after transplanting (DAT)."""
    if dat < 0:
        return "nursery"
    if dat <= 15:
        return "transplanting"
    if dat <= 45:
        return "tillering"
    if dat <= 65:
        return "panicle initiation"
    if dat <= 80:
        return "flowering"
    if dat <= 105:
        return "grain filling"
    return "maturity"


# ---------------------------------------------------------------- weather
SAMPLE_WEATHER = json.loads((DATA / "sample_weather.json").read_text(encoding="utf-8"))
DISTRICTS = list(SAMPLE_WEATHER["districts"].keys())


def get_weather(district: str, use_live: bool = True) -> dict:
    """Return a 5-day forecast. Tries the free Open-Meteo API, falls back to bundled sample data."""
    d = SAMPLE_WEATHER["districts"][district]
    if use_live:
        try:
            r = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": d["lat"], "longitude": d["lon"], "timezone": "Asia/Kolkata",
                    "forecast_days": 5,
                    "daily": "temperature_2m_max,temperature_2m_min,relative_humidity_2m_max,precipitation_sum",
                },
                timeout=6,
            )
            r.raise_for_status()
            j = r.json()["daily"]
            days = [
                {"date": j["time"][i], "tmax": j["temperature_2m_max"][i], "tmin": j["temperature_2m_min"][i],
                 "rh": j["relative_humidity_2m_max"][i], "rain": j["precipitation_sum"][i] or 0.0}
                for i in range(len(j["time"]))
            ]
            if days and all(x["tmax"] is not None for x in days):
                return {"source": "live (Open-Meteo)", "days": days}
        except Exception:
            pass
    today = date.today()
    days = [
        {"date": str(today + timedelta(days=i)), "tmax": v[0], "tmin": v[1], "rh": v[2], "rain": v[3]}
        for i, v in enumerate(d["sample"])
    ]
    return {"source": "sample data (offline)", "days": days}


def weather_advisories(wx: dict, stage: str) -> list[dict]:
    """Deterministic agronomy rules. These act as an expert 'rule of thumb' cross-check on the AI."""
    days = wx["days"]
    rain48 = sum(d["rain"] for d in days[:2])
    rain5 = sum(d["rain"] for d in days)
    tmax = max(d["tmax"] for d in days)
    rh_avg = sum(d["rh"] for d in days) / len(days)
    tmin_avg = sum(d["tmin"] for d in days) / len(days)
    out = []

    if rain48 >= 10:
        out.append({"level": "warn", "rule": "R1",
                    "text": f"{rain48:.0f} mm rain expected in next 48 h - postpone pesticide spray and urea top-dressing."})
    if rain5 >= 30:
        out.append({"level": "warn", "rule": "R2",
                    "text": f"Heavy rain ({rain5:.0f} mm over 5 days) - keep drainage open, strengthen bunds; BLB can spread with flood water."})
    if stage in ("tillering", "panicle initiation", "flowering") and rh_avg >= 85 and 20 <= tmin_avg <= 27:
        out.append({"level": "warn", "rule": "R3",
                    "text": "Humid weather with mild nights - high risk for blast and sheath blight. Scout fields every 3-4 days."})
    if stage in ("panicle initiation", "flowering", "grain filling") and rh_avg >= 80 and tmax <= 34:
        out.append({"level": "info", "rule": "R4",
                    "text": "Warm-humid conditions favour brown planthopper - tap the base of plants to check for hoppers."})
    if stage == "flowering" and tmax >= 35:
        out.append({"level": "warn", "rule": "R5",
                    "text": f"Max temperature {tmax:.0f} deg C at flowering - risk of spikelet sterility; keep a thin layer of water in the field."})
    if stage == "flowering" and rain5 >= 10 and rh_avg >= 85:
        out.append({"level": "info", "rule": "R6",
                    "text": "Rain + high humidity at flowering - watch for false smut."})
    if rain5 < 2 and tmax >= 36 and stage not in ("maturity",):
        out.append({"level": "info", "rule": "R7",
                    "text": "Hot and dry week - check field water; irrigate as per AWD (about 2 days after ponded water disappears)."})
    if stage == "maturity":
        out.append({"level": "info", "rule": "R8",
                    "text": "Near harvest - stop irrigation 10-15 days before harvest; plan straw management (no burning)."})
    if not out:
        out.append({"level": "ok", "rule": "R0", "text": "No weather-related risk flagged for this stage in the next 5 days."})
    return out


# ---------------------------------------------------------------- guardrails
INJECTION = re.compile(
    r"(ignore|disregard|forget)\s+(all|any|the|your|previous|above|earlier)?\s*(instructions|rules|prompt)"
    r"|system\s*prompt|you\s+are\s+now|act\s+as\s+(?!a\s+farmer)|jailbreak|\bDAN\b|developer\s+mode"
    r"|reveal\s+(your|the)\s+(prompt|instructions)",
    re.I,
)
EMERGENCY = re.compile(r"poison|swallow|drank|zehar|ज़हर|जहर|unconscious|behosh|बेहोश|vomit", re.I)
MAX_CHARS = 1200


def check_input(text: str) -> dict:
    """Pre-model validation. Returns {'ok': bool, 'reason': str, 'emergency': bool, 'text': cleaned}."""
    t = (text or "").strip()
    if not t:
        return {"ok": False, "reason": "empty", "emergency": False, "text": t}
    if len(t) > MAX_CHARS:
        return {"ok": False, "reason": "too_long", "emergency": False, "text": t[:MAX_CHARS]}
    if INJECTION.search(t):
        return {"ok": False, "reason": "injection", "emergency": False, "text": t}
    return {"ok": True, "reason": "", "emergency": bool(EMERGENCY.search(t)), "text": t}


GENERIC = {"india", "north", "west", "general", "best", "time", "days", "high", "where", "get", "much"}


GENERAL_Q = re.compile(r"this week|today|right now|what (should|do|can) i do|next step|is hafte|aaj kya|kya karu|kya karein|इस हफ्ते|क्या करूँ|क्या करें", re.I)


def in_domain(query: str) -> bool:
    """True if the query shares at least one title/keyword term with some KB entry.
    (Edge case found in testing: 'best cricket team in india' scored >0 via generic words in body text.)"""
    q = set(_tokens(query)) - GENERIC
    return any(q & set(_tokens(e["title"] + " " + e["keywords"])) for e in KB)


def offline_answer(query: str, stage: str) -> str:
    """Fallback when Gemini is unavailable: return the best-matching knowledge base entries verbatim."""
    # General "what now?" questions have no pest/nutrient keyword, so answer from the current crop stage.
    # (Edge case found in live testing: "What should I do in my field this week?" was refused offline.)
    if GENERAL_Q.search(query) and not in_domain(query):
        stage_notes = [e for e in KB if e["stage"] == stage][:2] or [e for e in KB if e["stage"] == "all"][:1]
        parts = [f"**{e['title']}** ({e['id']})\n\n{e['text']}" for e in stage_notes]
        return (f"For your current stage (**{stage}**), the key notes are below. Also check the "
                f"**Weather advisory** tab for this week's spray/irrigation alerts.\n\n" + "\n\n---\n\n".join(parts))
    hits = retrieve(query, k=2, stage=stage)
    if not hits or not in_domain(query):
        return ("I could not find this in my rice knowledge base. I can only help with rice cultivation "
                "(nursery, fertiliser, water, pests, diseases, weather, harvest). For other questions please "
                "contact your KVK or the Kisan Call Centre at 1800-180-1551.")
    top = hits[0][1]
    hits = [h for h in hits if h[1] >= 0.5 * top]      # drop weak second matches
    parts = [f"**{e['title']}** ({e['id']})\n\n{e['text']}" for e, _ in hits]
    return "\n\n---\n\n".join(parts)
