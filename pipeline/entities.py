"""Stage 2: transcript.json -> entities.json (places, routes, numbers, dates, key phrases, b-roll ideas), via Gemini.

    python -m pipeline.entities work/<episode>/transcript.json --out work/<episode>/entities.json [--model gemini-3.8-flash]

How timing stays exact: the transcript goes to the model as numbered words ("[12]गुड़गाउ [13]में ..."). The model
answers with word indices, never times; this script turns indices into start/end seconds from the transcript and
checks that the quoted text really is at those indices (fuzzy, so "11 .7" vs "11.7" passes). Anything that doesn't
line up is repaired by searching nearby words, or dropped with a warning.

Responses are cached in work/<episode>/cache/ by (transcript words, prompt version, model), so re-running costs nothing.
Requires GEMINI_API_KEY in .env.
"""
import argparse
import difflib
import re
import sys
import unicodedata
from pathlib import Path

from pipeline.common import read_json, require_env, sha, write_json

DEFAULT_MODEL = "gemini-3.8-flash"
PROMPT_VERSION = 6

PLACE_KINDS = ["country", "state", "district", "city", "neighbourhood", "river", "lake", "sea", "ocean", "mountain",
               "mountain_range", "region", "desert", "island", "continent", "landmark", "road", "highway",
               "metro_line", "metro_station", "railway_line", "railway_station", "airport", "port", "border"]
ROUTE_MODES = ["metro", "road", "rail", "river", "flight", "sea", "border", "walk", "other"]

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string", "description": "One English sentence: what the reel explains."},
        "main_place": {"type": "string", "description": "id of the entity the whole reel is about (usually a place)."},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "e1, e2, ... in order of first mention"},
                    "type": {"type": "string", "enum": ["place", "route", "number", "date", "phrase", "person", "thing"]},
                    "w0": {"type": "integer", "description": "index of the first word of the mention"},
                    "w1": {"type": "integer", "description": "index of the last word of the mention (inclusive)"},
                    "quote": {"type": "string", "description": "the mentioned words exactly as in the transcript"},
                    "name_en": {"type": "string", "description": "place/route: canonical English name, e.g. 'Gurugram', 'Delhi Metro Pink Line', 'NH 44'"},
                    "kind": {"type": "string", "enum": PLACE_KINDS},
                    "country": {"type": "string", "description": "ISO 3166-1 alpha-2 country of the place, e.g. IN"},
                    "parent": {"type": "string", "description": "containing state/city in English, e.g. 'Haryana'"},
                    "lat": {"type": "number", "description": "approximate latitude from your knowledge (used only to disambiguate)"},
                    "lon": {"type": "number"},
                    "mode": {"type": "string", "enum": ROUTE_MODES},
                    "stops": {"type": "array", "items": {"type": "string"},
                              "description": "route: ordered ids of place entities or English names of endpoints/stops mentioned"},
                    "value": {"type": "number", "description": "number: numeric value (11.7, 45, 1280000000)"},
                    "unit": {"type": "string", "description": "number: unit as shown, e.g. 'KM', 'Stations', '%', 'Billion'"},
                    "display": {"type": "string", "description": "how it should appear on screen, short, English/Latin script, e.g. '11.7 KM', '45 Stations', 'Richest & only tax-free state'"},
                    "refers_to": {"type": "string", "description": "number/date/phrase: id of the entity it describes"},
                    "emphasis": {"type": "string", "enum": ["title", "stat", "label", "keyword", "callout"]},
                    "visual": {"type": "string", "enum": ["highlight", "zoom", "route", "pin", "label", "stat", "title", "broll", "none"]},
                    "broll_query": {"type": "string", "description": "thing/person: English stock-footage search query (no real people's names)"},
                },
                "required": ["id", "type", "w0", "w1", "quote", "visual"],
            },
        },
        "sections": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "w0": {"type": "integer"}, "w1": {"type": "integer"},
                "role": {"type": "string", "enum": ["hook", "explain", "opinion", "cta"]}},
                "required": ["w0", "w1", "role"]},
        },
    },
    "required": ["topic", "entities", "sections"],
}

INSTRUCTIONS = """You are the research assistant for an Instagram map-explainer reel (Hindi/Hinglish/English voiceover).
The editor turns your output into map animations: each place gets highlighted or zoomed to when it is spoken, routes
draw between places, numbers pop up as stats. Read the numbered transcript and list every entity a viewer should SEE.

Rules:
- Indices: w0/w1 are the [n] word numbers in the transcript; quote = those words copied exactly. Use the FIRST word
  where the thing is named (graphics land on that word). Mention the same place again later as a new entity only if
  the narration comes back to it (the camera returns there).
- Places: every country, state, city, area, river, sea, mountain, landmark, road, metro line or station that is named.
  name_en is the standard English name people would search a map for; for Indian names written in Devanagari give the
  usual English spelling (गुड़गाउ -> Gurugram, द्वार्का -> Dwarka). Always fill kind, country, parent when known, and an
  approximate lat/lon. Use India's official names (Survey of India): Jammu & Kashmir, Ladakh, etc.
  For places and routes, display = the on-map label, the way the narrator said it, in Latin script and short
  ("मिलेनियम सिटी" -> "Millennium City", not "Millennium City Centre Gurugram Metro Station"; "ओल्ड गुड़गांव" ->
  "Old Gurgaon").
- Routes: lines, highways, rivers-as-paths, "from A to B" journeys, borders between two places. stops = the ids of the
  place entities the route passes through, in travel order, INCLUDING places named later in the narration as being on
  this route (e.g. "the new metro will go to Old Gurgaon, Palam Vihar, Udyog Vihar, Cyber City" -> those ids are the
  stops of that metro route). Use English names for stops that are not separate entities. mode = metro/road/rail/...
  Also give the route's approximate centre as lat/lon, and country.
- Numbers: every statistic (length, count, area, population, money, percentage, rank, year-as-number). Speech splits
  decimals ("11 .7", "71 .5 किलो") - join them. display is short and in Latin script as it would be typed on screen:
  "11.7 KM", "45 Stations", "₹5.8 Lakh", "1.28 Billion". refers_to = the entity it describes.
- Dates/years -> type date, display e.g. "1975".
- Phrases: the 1-3 strongest claims or superlatives worth typing on screen ("India's longest highway", "World's first
  organic state"), emphasis title/callout, display short.
- Keywords: short facts about a subject that an editor would type next to its stats, 1-2 words, e.g. "saare elevated
  hain" -> display "Elevated"; "24/7 operational" -> "24/7 Operational"; "underground". type phrase, emphasis keyword,
  refers_to = the subject, w0/w1 on the fact's key word.
- People: real people mentioned -> type person, visual none (we never generate real people). Things that cannot be
  shown on a map (trains, farms, food) -> type thing, visual broll, with an English broll_query.
- sections: split the whole transcript (every word index exactly once, in order) into hook (first claim), explain,
  opinion (the narrator's personal take, often at the end, usually on camera), cta (asks viewers to comment/follow).
- Do not invent entities that are not in the transcript. Do not translate quotes."""


def _norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    return re.sub(r"[\s.,!?।'\"()\-–—]+", "", s)


def numbered(words):
    return " ".join(f"[{i}]{w['w']}" for i, w in enumerate(words))


def call_gemini(model, transcript_text, api_key):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=model,
        contents=[INSTRUCTIONS, "TRANSCRIPT (numbered words):\n" + transcript_text],
        config=types.GenerateContentConfig(
            temperature=0.0, response_mime_type="application/json", response_json_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_level="medium")))
    return resp.text


def align(ent, words):
    """Check that quote is at [w0..w1]; if not, look for the best fuzzy match within ±15 words. Returns ok flag."""
    n = len(words)
    q = _norm(ent.get("quote", ""))
    if not q:
        return False

    def score(a, b):
        return difflib.SequenceMatcher(None, q, _norm("".join(w["w"] for w in words[a:b + 1]))).ratio()

    w0, w1 = max(0, min(ent["w0"], n - 1)), max(0, min(ent["w1"], n - 1))
    if w1 < w0:
        w0, w1 = w1, w0
    best = (score(w0, w1), w0, w1)
    if best[0] < 0.8:
        span = w1 - w0
        for a in range(max(0, w0 - 15), min(n, w0 + 16)):
            for s in range(max(0, span - 2), span + 3):
                b = min(n - 1, a + s)
                r = score(a, b)
                if r > best[0]:
                    best = (r, a, b)
    ent["w0"], ent["w1"], ent["match"] = best[1], best[2], round(best[0], 2)
    return best[0] >= 0.6


def extract(transcript, model=DEFAULT_MODEL, cache_dir=None):
    words = transcript["words"]
    text = numbered(words)
    key = sha([w["w"] for w in words], PROMPT_VERSION, model)
    cache = Path(cache_dir) / f"entities_{key}.json" if cache_dir else None
    if cache and cache.exists():
        raw = cache.read_text(encoding="utf-8")
        print(f"[entities] cached response {cache.name}")
    else:
        api_key = require_env("GEMINI_API_KEY", "Create a key at https://aistudio.google.com/apikey and put "
                                                "GEMINI_API_KEY=... in the .env file in the project folder.")
        raw = call_gemini(model, text, api_key)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(raw, encoding="utf-8")
    import json
    data = json.loads(raw)
    import jsonschema
    jsonschema.validate(data, SCHEMA)

    kept, dropped = [], []
    for e in data["entities"]:
        if not align(e, words):
            dropped.append(e)
            continue
        if e["type"] == "number":
            # a stat lands on the number itself ("टोटल 11 मेट्रो स्टेशन्स" -> on "11"), not on a lead-in word
            digit = next((i for i in range(e["w0"], e["w1"] + 1) if re.search(r"\d", words[i]["w"])), None)
            if digit is not None:
                e["w0"] = digit
        e["start"], e["end"] = words[e["w0"]]["start"], words[e["w1"]]["end"]
        kept.append(e)
    for s in data.get("sections", []):
        s["start"] = words[max(0, min(s["w0"], len(words) - 1))]["start"]
        s["end"] = words[max(0, min(s["w1"], len(words) - 1))]["end"]
    for e in dropped:
        print(f"[entities] WARNING dropped {e['id']} '{e.get('quote')}': not found near words {e['w0']}-{e['w1']}",
              file=sys.stderr)
    return {"transcript": transcript.get("audio"), "model": model, "prompt_version": PROMPT_VERSION,
            "topic": data["topic"], "main_place": data.get("main_place"),
            "entities": sorted(kept, key=lambda e: e["start"]), "sections": data.get("sections", []),
            "dropped": dropped}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()
    out = Path(a.out)
    res = extract(read_json(a.transcript), a.model, cache_dir=out.parent / "cache")
    write_json(out, res)
    counts = {}
    for e in res["entities"]:
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"[entities] {counts}, {len(res['dropped'])} dropped -> {out}")


if __name__ == "__main__":
    main()
