"""Story mode, stage 3b: transcript + entities -> storyboard.json (what is on screen, beat by beat), via Gemini.

    python -m pipeline.storyboard --episode work/<ep> [--model gemini-3.1-pro-preview] [--facecam]

The model plays the editor of the @adarshh.gupta reels (style/tokens.json "story"): it splits the narration into
beats of 2.5-6 s and picks for each one a visual (AI photo, real news clipping, map, face-cam, CTA paper, Follow card),
the on-screen text and the transition into it. Like entities.py it answers with word indices, never times; this script
checks every index, makes the beats tile the whole transcript, and turns indices into seconds.

    storyboard.json: {"topic", "beats": [{"id", "w0", "w1", "start", "end", "visual", "note", "image_prompt",
                      "character", "motion", "map": {...}, "news": {...}, "text": {...}, "transition", "sfx"}],
                      "characters": {"c1": "description"}, "cta_highlight": [word indices]}

Hand-edit it (change a prompt, swap a visual) and re-run from `visuals`.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from pipeline.common import read_json, require_env, sha, write_json

DEFAULT_MODEL = "gemini-3.1-pro-preview"
PROMPT_VERSION = 7
VISUALS = ["image", "map", "news", "graphic", "facecam", "cta", "follow"]
PRESETS = ["impact", "question", "chapter", "stat", "name", "date", "caption_box", "headline_top", "none"]
TRANSITIONS = ["fade", "blur", "zoom", "whip", "flash", "cloud", "glitch", "cut"]
MOTIONS = ["push_in", "pull_out", "pan_left", "pan_right", "rise", "sink"]

SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {"type": "string"},
        "characters": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "description": {"type": "string"}}, "required": ["id", "description"]}},
        "beats": {"type": "array", "items": {"type": "object", "properties": {
            "w0": {"type": "integer"}, "w1": {"type": "integer"},
            "visual": {"type": "string", "enum": VISUALS},
            "note": {"type": "string", "description": "one line, English: what the viewer sees"},
            "image_prompt": {"type": "string"},
            "character": {"type": "string", "description": "id of a recurring character shown in this image, if any"},
            "motion": {"type": "string", "enum": MOTIONS},
            "map": {"type": "object", "properties": {
                "places": {"type": "array", "items": {"type": "string"}, "description": "entity ids of places to colour/label"},
                "focus": {"type": "string", "description": "entity id the camera ends on"},
                "start_wide": {"type": "string", "enum": ["world", "continent", "country", "none"]},
                "style": {"type": "string", "enum": ["fill", "heat", "districts", "pins", "route", "photo", "flow"]},
                "photo_prompt": {"type": "string", "description": "style photo: the scene seen through the place's outline"},
                "flows": {"type": "array", "items": {"type": "object", "properties": {
                    "from": {"type": "string", "description": "entity id of the river/canal/route, or of the place it comes from"},
                    "label": {"type": "string", "description": "big line of its info box, e.g. '57% WATER'"},
                    "sub": {"type": "string", "description": "small line, e.g. 'FROM HARYANA (RAVI-BEAS + YAMUNA)'"},
                    "w": {"type": "integer"}}, "required": ["from", "label", "w"]}},
                "tilt": {"type": "boolean"},
                "callouts": {"type": "array", "items": {"type": "object", "properties": {
                    "place": {"type": "string"}, "label": {"type": "string"}, "w": {"type": "integer"}},
                    "required": ["place", "label", "w"]}}},
                "required": ["places", "focus", "style"]},
            "news": {"type": "object", "properties": {
                "query": {"type": "string", "description": "English search query for a real article about this claim"},
                "phrases": {"type": "array", "items": {"type": "string"},
                            "description": "1-2 short English phrases (2-6 words) likely printed in such an article, to highlight"},
                "w": {"type": "integer", "description": "word where the highlight sweeps in"},
                "photo": {"type": "boolean", "description": "keep the article's own photo on the card (a real person)"},
                "more": {"type": "array", "items": {"type": "string"}, "description": "montage: 1-2 more article queries"}},
                "required": ["query", "phrases", "w"]},
            "text": {"type": "object", "properties": {
                "preset": {"type": "string", "enum": PRESETS},
                "lines": {"type": "array", "items": {"type": "object", "properties": {
                    "text": {"type": "string"}, "role": {"type": "string"},
                    "w": {"type": "integer", "description": "word index where this line appears"},
                    "count_to": {"type": "number", "description": "stat number lines: the value it counts up to"}},
                    "required": ["text", "role", "w"]}}},
                "required": ["preset", "lines"]},
            "graphic": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": ["connect", "list"]},
                "items": {"type": "array", "items": {"type": "object", "properties": {
                    "text": {"type": "string"}, "w": {"type": "integer"}}, "required": ["text", "w"]}}},
                "required": ["kind", "items"]},
            "stamp": {"type": "object", "properties": {"text": {"type": "string"}, "w": {"type": "integer"}},
                      "required": ["text", "w"]},
            "transition": {"type": "string", "enum": TRANSITIONS},
            "sfx": {"type": "string", "enum": ["whoosh", "boom", "riser", "none"]}},
            "required": ["w0", "w1", "visual", "note", "transition"]}},
        "cta_highlight": {"type": "array", "items": {"type": "integer"},
                          "description": "word indices of the CTA question's key words (shown in red caps)"},
    },
    "required": ["topic", "beats"],
}

def _required_nullable(schema):
    """Every beat must answer image_prompt / map / news / graphic / text (null where it does not apply): with them
    optional, the model started leaving them all out once the schema grew (2026-10-23, ER storyboard)."""
    item = schema["properties"]["beats"]["items"]
    for k in ("image_prompt", "map", "news", "graphic", "text", "stamp"):
        item["properties"][k] = {"anyOf": [item["properties"][k], {"type": "null"}]}
    item["required"] = list(item["required"]) + ["image_prompt", "map", "news", "graphic", "text"]


_required_nullable(SCHEMA)

INSTRUCTIONS = """You are the editor of a top Indian explainer page on Instagram (Hindi/Hinglish voiceover, 9:16 reels,
style of @adarshh.gupta). You turn the narration into a storyboard. The finished reel changes picture every 2.5-6
seconds, never shows the same thing twice in a row, and every picture shows EXACTLY what is being said at that moment.

INPUT: numbered words with start times: [index|seconds]word. A list of entities (places with ids, numbers, phrases).

OUTPUT beats that cover every word exactly once, in order (beat k+1 starts at the word after beat k ends). Beat length
2.5-6 s (a CTA or Follow beat may be 2-8 s). Start a new beat where the idea changes.

VISUALS (pick per beat):
- "map": when a place, region, border, river or route is the point (where something is, how big, what is next to
  what). Use the entity ids from the list (only places). style: fill (colour the places), heat (a hot spot fading out,
  e.g. 'red shows where water dropped most'), districts (several sub-areas each in its own colour), pins (cities as
  labels), route. start_wide = how far out the camera starts before it dives in (first map of the reel: world or
  continent looks great). tilt = true for a dramatic 3D view. callouts = round magnifiers on spots ('AVALANCHE SITE').
  Do not use more than 2 map beats in a row. Places listed one after another ('France, Chile, Spain, India') stay on
  ONE map beat that covers the whole list (each fills on its own word), even if it runs 6-8 s.
  style photo: the place's outline cut into the map like a window, a picture of the problem inside it (Delhi's shape
  with an old dry tap over cracked earth inside: 'Delhi's water crisis'). photo_prompt = that scene (40-70 words,
  the object big and centred). Great for the reel's key place when the problem itself is the point.
  style flow: water / goods / people moving INTO the focus place along rivers, canals, pipelines or routes, glowing
  and flowing, each with an info box: flows = [{from: river/route/place entity id, label: '57% WATER', sub: 'FROM
  HARYANA (RAVI-BEAS + YAMUNA)', w}]. Use it whenever the narration says where something comes from and how much.
  Every reel uses style photo once for its main place (early, right after the opening).
- "news": when the narration cites a report, survey, government decision, scheme launch, court order, data release
  or a news event. query = what a journalist's article about it would be titled (English). phrases = short phrases
  that such an article would literally contain (numbers, names), to highlight. When a real person is named (a chief
  minister, a minister, an official, a company head) the beat should be a news beat with photo = true: the article's
  own photo shows them for real (we never generate real people).
  NEWS FIRST: every factual claim about real events, history, governments, agencies or companies (who did what, what
  a report found, what was revealed) is a news beat, not an AI picture - real sources make the reel credible. In a
  news-driven reel aim for 30-45 % news beats. When the narration piles up evidence ('kuch aur screenshots attach kar
  raha hoon', 'bahut baar ho chuka hai', several cases) make it a MONTAGE: more = 1-2 extra queries for other
  articles on the same point, shown flying in one on top of the other.
- "graphic": simple motion graphics on a dark studio background when an idea is abstract (no photo shows it):
  graphic.kind = connect (things being linked: 'bits and pieces ko join karna', a chain of events; 3-6 items, each a
  2-4 word card appearing on its word) or list (3-5 points / reasons / steps, each on its word).
- stamp (optional on an image or news beat): a rubber stamp slamming onto it on a word: 'DECLASSIFIED', 'CONFIRMED',
  'BANNED', 'FAKE' - at most twice per reel.
- "image": everything else: a cinematic AI photo that shows the sentence literally. image_prompt in English, 40-90
  words: subject, action, place (Indian context when the story is in India: Indian streets, people, clothing,
  vehicles, signage-free), camera (low angle / aerial / close-up / wide), light (golden hour, overcast, neon night,
  dramatic), mood, 'photorealistic, cinematic, shallow depth of field, 9:16 vertical'. When the sentence is about a
  city or country, put its best-known landmark in the scene so the viewer knows where we are without any text (Delhi:
  India Gate or Qutub Minar in the background; Mumbai: Gateway of India / Sea Link; Kolkata: Howrah Bridge; Paris:
  Eiffel Tower; Washington: the Capitol). Explainer diagrams are welcome
  as images too: 'isometric 3D illustration of ...', 'cross-section of soil layers showing ...', 'a dramatic 3D
  rendered map-like scene of ...'. NEVER a real, identifiable person (no politicians, celebrities, officials, the
  narrator): show them symbolically (an empty podium with flags, a silhouette, hands signing a document, the
  building they work in). No text, letters, logos or flags with writing in the image (text is added by the editor).
  A person the story follows (a shopkeeper, a farmer) is a recurring character: give them an id in characters with
  a fixed look (age, face, hair, clothes) and set character on every beat that shows them.
- "facecam": the narrator on camera (only if FACECAM is available, see below): use it for the narrator's own
  opinion/question and for 1-2 short beats in the middle where he talks to the viewer directly.
- "cta": the closing question to the audience ('comment karke batao ...'): torn paper card with the question.
  Put the key words of the question in cta_highlight (e.g. 'GOVERNMENT POLICY', 'RAINWATER HARVESTING SYSTEM').
- "follow": the final 'follow karein ...' line: Instagram profile Follow card.

TEXT on a beat (optional, the editor's overlay; English, Latin script, SHORT):
- impact: 2 lines, lead (white) + accent (yellow; roles: accent, accent_orange, accent_red): a hook or a key reveal,
  e.g. 'WHAT IS THE' / 'REASON ?', 'RAINWATER' / 'HARVESTING'. Max 3 words per line.
- question: the opening hook question, roles small / big_red / small: 'Where did this' / 'DEVASTATING FLOOD' /
  'in Nepal suddenly come from?'.
- chapter: roles kicker / word / sub: 'SECOND REASON' / 'PARALI'.
- stat: a number moment, roles body / num / body: 'Delhi receives' / '80%' (count_to 80) / 'of its annual rainfall
  during the monsoon months.' Body lines are normal sentence case (not caps) and read as one sentence with the number.
- name: a person the story follows, roles first / last ('Mudassir' / 'Naqshbandi').
- date: roles big / small: '26' / 'AUGUST'.
- headline_top: over a map, roles lead / accent: "INDIA'S 10 MOST" / 'POLLUTED CITIES'.
- caption_box: role text: one sentence of the narration in English, typed in a box (for a dramatic event).
Each line appears on its word (w = index of the word it belongs to). Not every beat needs text: at most 60 % of beats
have it, never 3 beats in a row, and never the impact preset on two beats in a row (it loses its punch).
Numbers spoken should almost always appear (stat preset, or a line in another preset).

TRANSITION into each beat: fade (default, calm), blur (into/out of news), zoom (dramatic reveal, into a map, after a
hook), whip (fast list of things), flash (a shock), cloud (into the first map from the sky), glitch (only into
follow), cut (rare). The first beat's transition is ignored. sfx: whoosh on zoom/whip, boom on a big reveal (2-4 per
reel), riser before the hook answer, else none.

Rules: indices exact; beats in order with no gaps or overlaps; English text on screen even when the voice is Hindi;
no invented facts on screen (numbers and names only as spoken)."""


def numbered(words):
    return " ".join(f"[{i}|{w['start']:.1f}]{w['w']}" for i, w in enumerate(words))


def entity_list(ents):
    out = []
    for e in ents["entities"]:
        d = {k: e.get(k) for k in ("id", "type", "kind", "name_en", "display", "w0") if e.get(k) is not None}
        out.append(d)
    return json.dumps(out, ensure_ascii=False)


def check(beats, words, facecam):
    """What breaks the format or the house rules in a storyboard (the model is asked to fix these)."""
    out = []
    for b in beats:
        a, z = words[b["w0"]]["start"], words[b["w1"]]["end"]
        lim = 8.5 if b["visual"] in ("cta", "follow") else 7.0
        if z - a > lim:
            out.append(f"beat at words {b['w0']}-{b['w1']} ({a:.1f}-{z:.1f} s) is {z - a:.1f} s long: split it into "
                       f"beats of 2.5-6 s, each with its own picture")
        if b["visual"] == "facecam" and isinstance(facecam, list):
            inside = max([min(z, q) - max(a, p) for p, q in facecam] + [0])
            if inside < 0.8 * (z - a):
                out.append(f"facecam beat {a:.1f}-{z:.1f} s is not inside the face-cam times {facecam}: end it where "
                           f"the face-cam time ends and show something else after")
        need = {"image": ("image_prompt",), "map": ("map", "places"), "news": ("news", "query"),
                "graphic": ("graphic", "items")}.get(b["visual"])
        if need and not (b.get(need[0]) if len(need) == 1 else (b.get(need[0]) or {}).get(need[1])):
            out.append(f"beat at words {b['w0']}-{b['w1']} is a {b['visual']} beat but has no {'.'.join(need)}: fill it")
        txt = f"{b.get('image_prompt', '')} {b.get('note', '')}".lower()
        if b["visual"] == "image" and any(k in txt for k in ("narrator", "the creator", "presenter", "talking to the camera",
                                                             "talking directly to the camera", "anchor")):
            out.append(f"beat {a:.1f}-{z:.1f} s: an image of the narrator is not allowed (a real person): show what he "
                       f"talks about instead")
    return out


def call(model, words, ents, facecam, previous=None, problems=None):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=require_env("GEMINI_API_KEY", "put GEMINI_API_KEY=... in .env"))
    if isinstance(facecam, list) and facecam:
        spans = ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in facecam)
        fc = (f"FACECAM: the narrator's clip can be shown ONLY inside these narration times: {spans}. A facecam beat "
              "must lie inside one of them; use them for his opinion lines and direct talk to the viewer.")
    elif facecam:
        fc = "FACECAM: available (the narrator's own clip covers the whole narration; use it for 15-30 % of the reel)"
    else:
        fc = "FACECAM: not available (never use the facecam visual)"
    resp = client.models.generate_content(
        model=model,
        contents=[INSTRUCTIONS, fc, f"ENTITIES: {entity_list(ents)}", "TRANSCRIPT:\n" + numbered(words)] + (
            ["YOUR PREVIOUS STORYBOARD:\n" + previous, "Fix these problems and answer with the whole corrected "
             "storyboard (same rules):\n- " + "\n- ".join(problems)] if previous else []),
        config=types.GenerateContentConfig(temperature=0.4, response_mime_type="application/json",
                                           response_json_schema=SCHEMA))
    return resp.text


def repair(beats, n_words, words):
    """Beats tile [0, n_words) in order: sort, clamp, close gaps (the earlier beat takes them), drop empty ones, and
    merge beats shorter than 1.2 s into a neighbour."""
    beats = sorted([b for b in beats if 0 <= b["w0"] < n_words], key=lambda b: b["w0"])
    out = []
    for b in beats:
        b["w1"] = max(b["w0"], min(b["w1"], n_words - 1))
        if out and b["w0"] <= out[-1]["w1"]:
            b["w0"] = out[-1]["w1"] + 1
            if b["w0"] > b["w1"]:
                continue
        if out and b["w0"] > out[-1]["w1"] + 1:
            out[-1]["w1"] = b["w0"] - 1
        out.append(b)
    if not out:
        raise SystemExit("storyboard: the model returned no usable beats")
    out[0]["w0"] = 0
    out[-1]["w1"] = n_words - 1
    merged = []
    for b in out:
        dur = words[b["w1"]]["end"] - words[b["w0"]]["start"]
        if merged and dur < 1.2 and b["visual"] not in ("cta", "follow"):
            merged[-1]["w1"] = b["w1"]
            continue
        merged.append(b)
    return merged


def house_rules(beats, ents, words):
    """The user's standing rules (memory: story-edit-feedback), enforced whatever the model chose:
    - a real person named during a news beat -> the card keeps the article's own photo of them;
    - the reel's main place gets one 'photo' map (its outline as a window onto the problem, q2 Delhi tap): the first
      early picture about that place becomes it, its prompt becoming the picture inside the outline."""
    people = [e for e in ents["entities"] if e["type"] == "person"]
    for b in beats:
        if b["visual"] == "news" and b.get("news") and any(b["start"] - 4 <= p["start"] <= b["end"] for p in people):
            b["news"]["photo"] = True
    main = next((e for e in ents["entities"] if e["id"] == ents.get("main_place")), None)
    if not main or any(b["visual"] == "map" and (b.get("map") or {}).get("style") == "photo" for b in beats):
        return
    names = {(main.get("name_en") or "").lower(), (main.get("display") or "").lower()} - {""}
    limit = words[-1]["end"] * 0.35
    for b in beats:
        text = f"{b.get('note', '')} {b.get('image_prompt', '')}".lower()
        if b["visual"] == "image" and b["start"] < limit and not b.get("character") and any(n in text for n in names):
            b["visual"] = "map"
            b["map"] = {"places": [main["id"]], "focus": main["id"], "style": "photo", "start_wide": "none",
                        "photo_prompt": b.get("image_prompt") or b.get("note", ""), "callouts": []}
            b["_house_rule"] = "photo map of the main place"
            break


def build(ep, model=DEFAULT_MODEL, facecam=False):
    ep = Path(ep)
    tr = read_json(ep / "transcript.json")
    ents = read_json(ep / "entities.json")
    words = tr["words"]
    key = sha([w["w"] for w in words], PROMPT_VERSION, model, facecam, [e["id"] for e in ents["entities"]])
    cache = ep / "cache" / f"storyboard_{key}.json"
    if cache.exists():
        raw = cache.read_text(encoding="utf-8")
        print(f"[storyboard] cached response {cache.name}")
    else:
        raw = call(model, words, ents, facecam)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(raw, encoding="utf-8")
    data = json.loads(raw)
    n = len(words)
    beats = repair(data["beats"], n, words)
    issues = check(beats, words, facecam)
    if issues:  # one round of self-correction: the model gets its own storyboard back with what is wrong in it
        print(f"[storyboard] fixing {len(issues)} problem(s): " + "; ".join(issues), file=sys.stderr)
        cache2 = ep / "cache" / f"storyboard_{key}_fix.json"
        if cache2.exists():
            raw2 = cache2.read_text(encoding="utf-8")
        else:
            raw2 = call(model, words, ents, facecam, previous=raw, problems=issues)
            cache2.write_text(raw2, encoding="utf-8")
        data = json.loads(raw2)
        beats = repair(data["beats"], n, words)
        left = check(beats, words, facecam)
        if left:
            print("[storyboard] still: " + "; ".join(left), file=sys.stderr)
    place_ids = {e["id"] for e in ents["entities"] if e["type"] in ("place", "route")}
    clampw = lambda w, b: max(b["w0"], min(int(w), b["w1"]))  # noqa: E731
    for i, b in enumerate(beats):
        b["id"] = f"b{i + 1:02d}"
        b["start"], b["end"] = words[b["w0"]]["start"], words[b["w1"]]["end"]
        if b["visual"] == "map":
            m = b.get("map") or {}
            m["places"] = [p for p in m.get("places", []) if p in place_ids]
            if m.get("focus") not in place_ids:
                m["focus"] = m["places"][0] if m["places"] else None
            for c in m.get("callouts", []):
                c["w"] = clampw(c["w"], b)
            for f in m.get("flows", []):
                f["w"] = clampw(f["w"], b)
            if not m["places"] and not m.get("focus"):
                print(f"[storyboard] {b['id']}: map beat with no known place -> image", file=sys.stderr)
                b["visual"] = "image"
                b.setdefault("image_prompt", b.get("note", ""))
            b["map"] = m
        if b.get("news"):
            b["news"]["w"] = clampw(b["news"].get("w", b["w0"]), b)
        for it in (b.get("graphic") or {}).get("items", []):
            it["w"] = clampw(it.get("w", b["w0"]), b)
            it["t"] = words[it["w"]]["start"]
        if b.get("stamp"):
            b["stamp"]["w"] = clampw(b["stamp"].get("w", b["w0"]), b)
            b["stamp"]["t"] = words[b["stamp"]["w"]]["start"]
        if b["visual"] == "graphic" and not (b.get("graphic") or {}).get("items"):
            b["visual"] = "image"
            b.setdefault("image_prompt", b.get("note", ""))
        t = b.get("text")
        if t and t.get("preset") not in (None, "none"):
            for ln in t["lines"]:
                ln["w"] = clampw(ln.get("w", b["w0"]), b)
                ln["t"] = words[ln["w"]]["start"]
        else:
            b.pop("text", None)
    if beats[0].get("transition"):
        beats[0]["transition"] = "cut"
    if isinstance(facecam, list):  # a face-cam beat outside the times the face is really on screen: a photo instead
        for b in beats:
            if b["visual"] != "facecam":
                continue
            inside = max([min(b["end"], z) - max(b["start"], a) for a, z in facecam] + [0])
            if inside < 0.8 * (b["end"] - b["start"]):
                print(f"[storyboard] {b['id']}: face-cam outside the face windows -> photo", file=sys.stderr)
                b["visual"] = "image"
                b.setdefault("image_prompt", b.get("note", ""))
    house_rules(beats, ents, words)
    # the narration ends on a statement with room after it (a remake whose borrowed ending was cut): our own outro,
    # the comment question on paper and the Follow card, over the music
    tail = tr["duration"] - words[-1]["end"]
    if tail >= 4 and not any(b["visual"] in ("cta", "follow") for b in beats):
        t0, t1 = round(words[-1]["end"] + 0.25, 3), round(tr["duration"] - 2.7, 3)
        last = len(words) - 1
        beats.append({"id": f"b{len(beats) + 1:02d}", "visual": "cta", "outro": True, "w0": last, "w1": last,
                      "start": t0, "end": t1, "note": "closing comment question (no voice)", "transition": "zoom",
                      "sfx": "riser"})
        beats.append({"id": f"b{len(beats) + 1:02d}", "visual": "follow", "outro": True, "w0": last, "w1": last,
                      "start": t1, "end": round(tr["duration"], 3), "note": "Follow card", "transition": "glitch",
                      "sfx": "whoosh"})
    res = {"model": model, "prompt_version": PROMPT_VERSION, "topic": data.get("topic"),
           "characters": {c["id"]: c["description"] for c in data.get("characters", [])},
           "cta_highlight": [w for w in data.get("cta_highlight", []) if 0 <= w < n], "beats": beats}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--facecam", action="store_true")
    a = ap.parse_args()
    res = build(a.episode, a.model, a.facecam)
    out = write_json(Path(a.episode) / "storyboard.json", res)
    kinds = {}
    for b in res["beats"]:
        kinds[b["visual"]] = kinds.get(b["visual"], 0) + 1
    print(f"[storyboard] {len(res['beats'])} beats {kinds} -> {out}")


if __name__ == "__main__":
    main()
