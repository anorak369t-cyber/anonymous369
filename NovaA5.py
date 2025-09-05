#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nova: all-in-one pure-Python offline STEM assistant
Features:
- NLU: tokenization, synonyms, fuzzy matching, intent detection, simple entity extraction
- Dialogue: multi-turn memory, pronoun resolution, adaptive tone/modes, clarifying questions
- Knowledge Base: CRUD, tags, confidence scoring, import/export, search with fuzzy + tags
- Memory: user profiles, sessions, transcripts
- Tools/Skills: math solver (safe), unit converter, time/date, dictionary, physics helper,
  chemistry helper (periodic table + molar mass), coding tutor explainer, quizzes, flashcards (SM-2),
  random facts
- Offline search index: simple inverted index over KB
- Storage: JSON files (portable), autosave
"""
import ast
import datetime as dt
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None  # Python <3.9 fallback

DATA_DIR = os.path.join(os.path.dirname(__file__), "nova_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------
# Utilities
# -------------------------------
def now_utc():
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)

def to_local(ts: dt.datetime, tzname: str):
    if tzname and ZoneInfo:
        try:
            return ts.astimezone(ZoneInfo(tzname))
        except Exception:
            return ts
    return ts

def slugify(s: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', s.lower()).strip('-')

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())

def tokenize(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", s.lower())

def levenshtein(a: str, b: str) -> int:
    # O(len(a)*len(b)) dynamic programming
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(
                dp[j] + 1,       # deletion
                dp[j-1] + 1,     # insertion
                prev + (0 if ca == cb else 1)  # substitution
            )
            prev = cur
    return dp[-1]

def fuzzy_ratio(a: str, b: str) -> float:
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 1.0
    dist = levenshtein(a, b)
    maxlen = max(1, max(len(a), len(b)))
    return 1.0 - (dist / maxlen)

def unique(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

# -------------------------------
# Storage (JSON-based)
# -------------------------------
class JSONStore:
    def __init__(self, path_base=DATA_DIR):
        self.paths = {
            "kb": os.path.join(path_base, "kb.json"),
            "synonyms": os.path.join(path_base, "synonyms.json"),
            "profiles": os.path.join(path_base, "profiles.json"),
            "sessions": os.path.join(path_base, "sessions.json"),
            "transcripts": os.path.join(path_base, "transcripts.json"),
            "dictionary": os.path.join(path_base, "dictionary.json"),
            "periodic": os.path.join(path_base, "periodic_table.json"),
            "facts": os.path.join(path_base, "facts.json"),
            "flashcards": os.path.join(path_base, "flashcards.json"),
        }
        self._cache = {k: self._load(v) for k, v in self.paths.items()}

        # Ensure defaults
        if not self._cache["synonyms"]:
            self._cache["synonyms"] = {
                "o-level": "uganda o level",
                "olevel": "uganda o level",
                "ole": "uganda o level",
                "sec school": "secondary school",
                "st": "science and technology",
            }
        if not self._cache["dictionary"]:
            self._cache["dictionary"] = {
                "photosynthesis": {"pos": "noun", "def": "Process by which plants use light to convert CO2 and water into glucose and oxygen.", "example": "Photosynthesis occurs in chloroplasts."},
                "atom": {"pos": "noun", "def": "Smallest unit of ordinary matter that forms a chemical element.", "example": "Atoms combine to form molecules."},
            }
        if not self._cache["periodic"]:
            self._cache["periodic"] = {
                "H": {"name": "Hydrogen", "atomic_mass": 1.008},
                "C": {"name": "Carbon", "atomic_mass": 12.011},
                "N": {"name": "Nitrogen", "atomic_mass": 14.007},
                "O": {"name": "Oxygen", "atomic_mass": 15.999},
                "Na": {"name": "Sodium", "atomic_mass": 22.990},
                "Cl": {"name": "Chlorine", "atomic_mass": 35.45},
                "K": {"name": "Potassium", "atomic_mass": 39.0983},
                "Ca": {"name": "Calcium", "atomic_mass": 40.078},
                "Fe": {"name": "Iron", "atomic_mass": 55.845},
                "Cu": {"name": "Copper", "atomic_mass": 63.546},
                "Zn": {"name": "Zinc", "atomic_mass": 65.38},
                "Ag": {"name": "Silver", "atomic_mass": 107.8682},
                "Au": {"name": "Gold", "atomic_mass": 196.96657},
            }
        if not self._cache["facts"]:
            self._cache["facts"] = [
                {"text": "Lightning heats air to temperatures hotter than the Sun’s surface.", "tags": ["physics", "weather"]},
                {"text": "Plants produce oxygen as a byproduct of photosynthesis.", "tags": ["biology"]},
                {"text": "Water is densest at about 4°C.", "tags": ["chemistry", "physics"]},
            ]
        self._autosave()

    def _load(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {} if path.endswith(".json") else {}

    def save(self, key):
        with open(self.paths[key], "w", encoding="utf-8") as f:
            json.dump(self._cache[key], f, ensure_ascii=False, indent=2)

    def _autosave(self):
        for k in self.paths.keys():
            self.save(k)

    def get(self, key):
        return self._cache[key]

    def set(self, key, value):
        self._cache[key] = value
        self.save(key)

# -------------------------------
# Data models
# -------------------------------
@dataclass
class KBEntry:
    id: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    source: str = "user"
    updated_at: float = field(default_factory=lambda: time.time())
    version: int = 1

@dataclass
class UserProfile:
    user_id: str
    name: str = "User"
    timezone: str = "Africa/Kampala"
    locale: str = "en"
    interests: List[str] = field(default_factory=lambda: ["STEM"])
    tone: str = "teacher"  # teacher|peer|quizmaster

@dataclass
class Session:
    session_id: str
    user_id: str
    context: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=lambda: time.time())

# -------------------------------
# Knowledge Base + Index
# -------------------------------
class KB:
    def __init__(self, store: JSONStore):
        self.store = store
        self.entries: Dict[str, KBEntry] = {}
        self._load()
        self._build_index()

    def _load(self):
        raw = self.store.get("kb")
        if isinstance(raw, list):
            # legacy list
            for e in raw:
                self.entries[e["id"]] = KBEntry(**e)
        elif isinstance(raw, dict):
            for _id, e in raw.items():
                self.entries[_id] = KBEntry(**e)
        else:
            self.entries = {}

    def _persist(self):
        self.store.set("kb", {k: asdict(v) for k, v in self.entries.items()})

    def _build_index(self):
        self.inv_index = defaultdict(set)
        self.tag_index = defaultdict(set)
        for eid, e in self.entries.items():
            text = f"{e.title} {e.content}"
            for tok in unique(tokenize(text)):
                self.inv_index[tok].add(eid)
            for t in e.tags:
                self.tag_index[t.lower()].add(eid)

    def add(self, title: str, content: str, tags: List[str], source: str = "user") -> KBEntry:
        eid = slugify(f"{title}-{int(time.time()*1000)}")
        entry = KBEntry(id=eid, title=title, content=content, tags=[t.lower() for t in tags], source=source)
        self.entries[eid] = entry
        self._persist()
        self._build_index()
        return entry

    def update(self, eid: str, **kwargs) -> Optional[KBEntry]:
        if eid not in self.entries: return None
        e = self.entries[eid]
        for k, v in kwargs.items():
            if hasattr(e, k):
                setattr(e, k, v)
        e.version += 1
        e.updated_at = time.time()
        self._persist()
        self._build_index()
        return e

    def forget(self, eid: str) -> bool:
        if eid in self.entries:
            del self.entries[eid]
            self._persist()
            self._build_index()
            return True
        return False

    def import_jsonl(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.add(obj["title"], obj["content"], obj.get("tags", []), obj.get("source", "import"))

    def export_jsonl(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            for e in self.entries.values():
                f.write(json.dumps(asdict(e)) + "\n")

    def _score(self, query: str, entry: KBEntry, tag_boost: float, recency_boost: float, source_weight: float):
        lex = max(
            fuzzy_ratio(query, entry.title),
            fuzzy_ratio(query, entry.content[:200])
        )
        tag = tag_boost if any(t in query.lower() for t in entry.tags) else 0.0
        recency = 1.0 / (1.0 + math.exp(-(entry.updated_at - (time.time()-86400))/86400))  # sigmoid recent=1
        return 0.6*lex + 0.2*tag + 0.15*recency + 0.05*source_weight

    def search(self, query: str, tags: Optional[List[str]] = None, top_k: int = 5):
        q_tokens = tokenize(query)
        candidate_ids = set()
        for t in q_tokens:
            candidate_ids |= self.inv_index.get(t, set())
        # If no lexical candidates, consider all
        if not candidate_ids:
            candidate_ids = set(self.entries.keys())
        if tags:
            tagset = set([t.lower() for t in tags])
            tagged = set()
            for t in tagset:
                tagged |= self.tag_index.get(t, set())
            if tagged:
                candidate_ids &= tagged
        scored = []
        for cid in candidate_ids:
            e = self.entries[cid]
            sweight = 1.0 if e.source == "nova_manual" else 0.8 if e.source == "user" else 0.7
            score = self._score(query, e, tag_boost=0.2, recency_boost=0.15, source_weight=sweight)
            scored.append((score, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

# -------------------------------
# NLU and Intent Detection
# -------------------------------
class NLU:
    def __init__(self, store: JSONStore):
        self.synonyms = store.get("synonyms")

        # Intent keyword sets
        self.intent_keywords = {
            "math": ["calculate", "solve", "evaluate", "math", "+", "-", "*", "/", "^"],
            "convert": ["convert", "to", "in", "km", "m", "cm", "kg", "g", "c", "f"],
            "time": ["time", "date", "today", "tomorrow", "now"],
            "define": ["define", "meaning", "what is", "explain", "definition"],
            "physics": ["force", "ohm", "current", "voltage", "velocity", "acceleration", "momentum", "work", "energy"],
            "chemistry": ["molar", "mass", "periodic", "element", "compound", "formula"],
            "code": ["python", "php", "loop", "function", "def", "class", "array", "dict", "bug"],
            "quiz": ["quiz", "test", "practice", "mcq", "question"],
            "flashcards": ["flashcard", "review", "remember", "spaced"],
            "kb": ["learn:", "save", "remember", "kb", "search kb", "forget"],
            "smalltalk": ["hello", "hi", "hey", "thanks", "thank you", "who are you"],
            "random_fact": ["random fact", "tell me a fact"],
        }

    def apply_synonyms(self, text: str) -> str:
        t = normalize_text(text)
        for ph, canon in self.synonyms.items():
            t = re.sub(rf"\b{re.escape(ph)}\b", canon, t)
        return t

    def detect_intent(self, text: str) -> str:
        t = text
        for intent, kws in self.intent_keywords.items():
            for kw in kws:
                if kw in t:
                    return intent
        return "kb" if len(t.split()) > 2 else "smalltalk"

    def extract_entities(self, text: str) -> Dict[str, Any]:
        # Very light entity extraction
        ents = {}
        # Units (length/mass/temp)
        m = re.search(r"([-+]?[0-9]*\.?[0-9]+)\s*([a-zA-Z°]+)\s*(?:to|in)\s*([a-zA-Z°]+)", text)
        if m:
            ents["value"] = float(m.group(1))
            ents["from_unit"] = m.group(2)
            ents["to_unit"] = m.group(3)
        # Math expression
        if any(sym in text for sym in ["+", "-", "*", "/", "^", "(", ")"]):
            ents["expr"] = text
        # Define target
        m2 = re.search(r"(?:define|what is|meaning of|explain)\s+([a-z0-9\s\-]+)\??", text)
        if m2:
            ents["term"] = m2.group(1).strip()
        return ents

    def analyze(self, text: str) -> Dict[str, Any]:
        expanded = self.apply_synonyms(text)
        intent = self.detect_intent(expanded)
        entities = self.extract_entities(expanded)
        return {"intent": intent, "entities": entities, "expanded": expanded}

# -------------------------------
# Dialogue Management
# -------------------------------
class DialogueManager:
    def __init__(self, profile: UserProfile, kb: KB):
        self.profile = profile
        self.kb = kb
        self.last_subject: Optional[str] = None

    def resolve_pronouns(self, utterance: str, context: List[Dict[str, Any]]) -> Optional[str]:
        # Simple heuristic: last subject mentioned in context
        if self.last_subject:
            if re.search(r"\b(it|this|that|they|them)\b", utterance):
                return self.last_subject
        # Try extract noun-ish subject from last turns
        for turn in reversed(context[-5:]):
            subj = turn.get("subject")
            if subj:
                return subj
        return None

    def set_subject(self, subject: Optional[str]):
        if subject:
            self.last_subject = subject

    def tone_wrap(self, content: str, mode: Optional[str] = None) -> str:
        mode = mode or self.profile.tone
        if mode == "teacher":
            return content
        if mode == "peer":
            return content
        if mode == "quizmaster":
            # Prompt with a follow-up question style
            return content + " Ready for a quick check?"
        return content

# -------------------------------
# Skills/Tools
# -------------------------------
class MathSolver:
    SAFE_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Pow,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.FloorDiv,
        ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Tuple, ast.List, ast.Constant,
        ast.BitXor  # for ^ used as power request; we remap
    }
    SAFE_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "exp": math.exp, "abs": abs}
    SAFE_NAMES = {"pi": math.pi, "e": math.e}

    def eval_expr(self, expr: str) -> float:
        expr = expr.replace("^", "**")
        node = ast.parse(expr, mode="eval")
        if not self._is_safe(node):
            raise ValueError("Unsafe expression")
        return self._eval(node.body)

    def _is_safe(self, node):
        for child in ast.walk(node):
            if type(child) not in self.SAFE_NODES:
                return False
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name) or child.func.id not in self.SAFE_FUNCS:
                    return False
        return True

    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            l = self._eval(node.left)
            r = self._eval(node.right)
            if isinstance(node.op, ast.Add): return l + r
            if isinstance(node.op, ast.Sub): return l - r
            if isinstance(node.op, ast.Mult): return l * r
            if isinstance(node.op, ast.Div): return l / r
            if isinstance(node.op, ast.FloorDiv): return l // r
            if isinstance(node.op, ast.Mod): return l % r
            if isinstance(node.op, ast.Pow): return l ** r
        if isinstance(node, ast.UnaryOp):
            v = self._eval(node.operand)
            if isinstance(node.op, ast.UAdd): return +v
            if isinstance(node.op, ast.USub): return -v
        if isinstance(node, ast.Name):
            if node.id in self.SAFE_NAMES: return self.SAFE_NAMES[node.id]
            raise ValueError("Unknown name")
        if isinstance(node, ast.Call):
            f = self.SAFE_FUNCS[node.func.id]
            args = [self._eval(a) for a in node.args]
            return f(*args)
        if isinstance(node, (ast.Tuple, ast.List)):
            return [self._eval(e) for e in node.elts]
        raise ValueError("Unsupported expression")

class UnitConverter:
    # Base units: m, g, s, C/F
    LENGTH = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0}
    MASS = {"mg": 0.001, "g": 1.0, "kg": 1000.0}
    TIME = {"ms": 0.001, "s": 1.0, "min": 60.0, "h": 3600.0}
    # Temperature handled separately
    def convert(self, value: float, from_u: str, to_u: str) -> str:
        fu, tu = from_u.lower(), to_u.lower()
        if fu in self.LENGTH and tu in self.LENGTH:
            base = value * self.LENGTH[fu]
            out = base / self.LENGTH[tu]
            return f"{value} {from_u} = {out:g} {to_u}"
        if fu in self.MASS and tu in self.MASS:
            base = value * self.MASS[fu]
            out = base / self.MASS[tu]
            return f"{value} {from_u} = {out:g} {to_u}"
        if fu in self.TIME and tu in self.TIME:
            base = value * self.TIME[fu]
            out = base / self.TIME[tu]
            return f"{value} {from_u} = {out:g} {to_u}"
        # Temperature
        if fu in ["c", "°c", "celsius"] and tu in ["f", "°f", "fahrenheit"]:
            out = (value * 9/5) + 32
            return f"{value} °C = {out:g} °F"
        if fu in ["f", "°f", "fahrenheit"] and tu in ["c", "°c", "celsius"]:
            out = (value - 32) * 5/9
            return f"{value} °F = {out:g} °C"
        raise ValueError("Unsupported units")

class TimeDate:
    def now_local(self, tzname: str) -> str:
        t = to_local(now_utc(), tzname)
        return t.strftime("%A, %d %B %Y, %H:%M:%S %Z")

class Dictionary:
    def __init__(self, store: JSONStore):
        self.data = store.get("dictionary")

    def define(self, term: str) -> Optional[str]:
        key = normalize_text(term)
        if key in self.data:
            d = self.data[key]
            return f"{term} ({d['pos']}): {d['def']} Example: {d['example']}"
        # fuzzy fallback
        best, score = None, 0.0
        for k in self.data.keys():
            r = fuzzy_ratio(key, k)
            if r > score:
                best, score = k, r
        if best and score > 0.6:
            d = self.data[best]
            return f"{best} ({d['pos']}): {d['def']} Example: {d['example']}"
        return None

class Physics:
    # Simple formula registry with solvers
    def solve_ohms_law(self, given: Dict[str, float]) -> Dict[str, Any]:
        # V = I * R
        res = {"formula": "V = I * R", "steps": []}
        I, R, V = given.get("I"), given.get("R"), given.get("V")
        if V is None and I is not None and R is not None:
            val = I * R
            res["V"] = val
            res["steps"] = ["V = I*R", f"V = {I}*{R}", f"V = {val} V"]
        elif I is None and V is not None and R is not None:
            val = V / R
            res["I"] = val
            res["steps"] = ["I = V/R", f"I = {V}/{R}", f"I = {val} A"]
        elif R is None and V is not None and I is not None:
            val = V / I
            res["R"] = val
            res["steps"] = ["R = V/I", f"R = {V}/{I}", f"R = {val} Ω"]
        return res

    def solve_motion(self, given: Dict[str, float]) -> Dict[str, Any]:
        # v = u + a t ; s = ut + 1/2 a t^2 (SUVAT subset)
        res = {"formula": "v = u + a t ; s = u t + 0.5 a t^2", "steps": []}
        u, a, t, v, s = given.get("u"), given.get("a"), given.get("t"), given.get("v"), given.get("s")
        if v is None and u is not None and a is not None and t is not None:
            vv = u + a*t
            res["v"] = vv
            res["steps"].append(f"v = u + a t = {u} + {a}*{t} = {vv}")
        if s is None and u is not None and a is not None and t is not None:
            ss = u*t + 0.5*a*t*t
            res["s"] = ss
            res["steps"].append(f"s = u t + 0.5 a t^2 = {u}*{t} + 0.5*{a}*{t}^2 = {ss}")
        return res

class Chemistry:
    def __init__(self, store: JSONStore):
        self.table = store.get("periodic")

    def molar_mass(self, formula: str) -> float:
        tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
        total = 0.0
        for sym, cnt in tokens:
            if sym not in self.table:
                raise ValueError(f"Unknown element: {sym}")
            n = int(cnt) if cnt else 1
            total += self.table[sym]["atomic_mass"] * n
        return total

class CodingTutor:
    def explain(self, code: str, lang_hint: Optional[str] = None) -> str:
        code_l = code.strip().lower()
        hints = []
        if "for " in code_l and " in " in code_l and ":" in code_l:
            hints.append("This is a Python for-loop iterating over an iterable.")
        if "def " in code_l and "(" in code_l and "):" in code_l:
            hints.append("Function definition with parameters; body is indented.")
        if "if " in code_l and ":" in code_l:
            hints.append("Conditional; runs the indented block when the condition is True.")
        if "<?php" in code or "$" in code and "echo" in code_l:
            hints.append("PHP snippet; echo outputs text; $var denotes a variable.")
        if not hints:
            hints.append("This code defines logic using basic control structures and variables.")
        return " ".join(hints)

class QuizEngine:
    def __init__(self, kb: KB):
        self.kb = kb
        self.history = []

    def generate(self, topic: Optional[str] = None, difficulty: str = "easy") -> Dict[str, Any]:
        # Use KB titles/content to create a simple question
        candidates = list(self.kb.entries.values())
        if topic:
            candidates = [e for e in candidates if topic.lower() in (e.title.lower() + " " + e.content.lower()) or topic.lower() in e.tags]
        if not candidates:
            # fallback
            q = {"question": "What is photosynthesis?", "type": "short", "answer": "Process by which plants make glucose using light, CO2, and water, releasing oxygen."}
            return q
        e = random.choice(candidates)
        if difficulty == "easy":
            q = {"question": f"Briefly define: {e.title}", "type": "short", "answer": e.content[:140] + ("..." if len(e.content) > 140 else "")}
        else:
            q = {"question": f"Explain one key idea from: {e.title}", "type": "short", "answer": e.content}
        return q

class Flashcards:
    # SM-2 simplified
    def __init__(self, store: JSONStore):
        self.store = store
        self.cards = store.get("flashcards") or {}

    def add(self, front: str, back: str, tags: Optional[List[str]] = None):
        cid = slugify(front)
        self.cards[cid] = {
            "front": front, "back": back, "tags": tags or [],
            "repetition": 0, "interval": 1, "easiness": 2.5,
            "due": time.time()
        }
        self.store.set("flashcards", self.cards)

    def review_queue(self, n=5):
        now = time.time()
        due = [(cid, c) for cid, c in self.cards.items() if c["due"] <= now]
        due.sort(key=lambda x: x[1]["due"])
        return due[:n]

    def update(self, cid: str, quality: int):
        c = self.cards[cid]
        q = max(0, min(5, quality))
        if q < 3:
            c["repetition"] = 0
            c["interval"] = 1
        else:
            if c["repetition"] == 0:
                c["interval"] = 1
            elif c["repetition"] == 1:
                c["interval"] = 6
            else:
                c["interval"] = round(c["interval"] * c["easiness"])
            c["repetition"] += 1
            c["easiness"] = max(1.3, c["easiness"] + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        c["due"] = time.time() + c["interval"] * 86400
        self.store.set("flashcards", self.cards)

class RandomFacts:
    def __init__(self, store: JSONStore):
        self.facts = store.get("facts")

    def get(self, topic: Optional[str] = None) -> str:
        pool = self.facts
        if topic:
            pool = [f for f in self.facts if topic.lower() in [t.lower() for t in f.get("tags", [])] or topic.lower() in f["text"].lower()]
        if not pool:
            pool = self.facts
        return random.choice(pool)["text"]

# -------------------------------
# Router and Reply
# -------------------------------
@dataclass
class Reply:
    text: str
    confidence: float = 1.0
    follow_up: Optional[str] = None
    sources: List[str] = field(default_factory=list)
    subject: Optional[str] = None

class Router:
    def __init__(self, store: JSONStore, profile: UserProfile, kb: KB):
        self.store = store
        self.profile = profile
        self.kb = kb
        self.nlu = NLU(store)
        self.dialogue = DialogueManager(profile, kb)
        self.math = MathSolver()
        self.units = UnitConverter()
        self.clock = TimeDate()
        self.dict = Dictionary(store)
        self.physics = Physics()
        self.chem = Chemistry(store)
        self.coding = CodingTutor()
        self.quiz = QuizEngine(kb)
        self.flash = Flashcards(store)
        self.facts = RandomFacts(store)

    def handle(self, text: str, session: Session) -> Reply:
        raw = text.strip()
        expanded = self.nlu.apply_synonyms(raw)
        analysis = self.nlu.analyze(raw)
        intent, ents = analysis["intent"], analysis["entities"]

        # Pronoun resolution
        subject = self.dialogue.resolve_pronouns(expanded, session.context)

        # Route by intent
        try:
            if intent == "convert" and "value" in ents:
                out = self.units.convert(ents["value"], ents["from_unit"], ents["to_unit"])
                rep = self._wrap(out, confidence=0.95, subject=f"convert {ents['from_unit']} to {ents['to_unit']}")
            elif intent == "math" and "expr" in ents:
                val = self.math.eval_expr(ents["expr"])
                rep = self._wrap(f"{val}", confidence=0.9, subject="math expression")
            elif intent == "time":
                out = self.clock.now_local(self.profile.timezone)
                rep = self._wrap(f"Local time: {out}", confidence=0.9, subject="time")
            elif intent == "define":
                term = ents.get("term") or raw.replace("define", "").strip()
                meaning = self.dict.define(term)
                if meaning:
                    rep = self._wrap(meaning, confidence=0.85, subject=term)
                else:
                    rep = self._wrap(f"I don't have a definition for '{term}'. Want me to save one if you provide it?", confidence=0.4)
            elif intent == "physics":
                rep = self._physics_route(expanded)
            elif intent == "chemistry":
                rep = self._chem_route(expanded)
            elif intent == "code":
                rep = self._wrap(self.coding.explain(raw), confidence=0.7, subject="code snippet")
            elif intent == "quiz":
                rep = self._quiz_route(expanded)
            elif intent == "flashcards":
                rep = self._flash_route(expanded)
            elif intent == "random_fact":
                rep = self._wrap(self.facts.get(), confidence=0.7, subject="fact")
            elif intent == "kb":
                rep = self._kb_route(expanded, subject_hint=subject)
            else:
                rep = self._smalltalk(expanded)
        except Exception as ex:
            rep = self._wrap(f"I hit a snag: {ex}", confidence=0.2)

        # Save context and subject
        session.context.append({"user": raw, "intent": intent, "entities": ents, "subject": rep.subject or subject})
        self.dialogue.set_subject(rep.subject or subject)
        return rep

    def _wrap(self, text, confidence=1.0, follow_up=None, subject=None, sources=None):
        content = self.dialogue.tone_wrap(text)
        return Reply(text=content, confidence=confidence, follow_up=follow_up, subject=subject, sources=sources or [])

    def _clarify(self, options: List[str]):
        return self._wrap("Do you mean: " + " / ".join(options) + "?", confidence=0.3)

    def _physics_route(self, text: str) -> Reply:
        # Heuristic parse: ohm, motion
        if "ohm" in text or "voltage" in text or "current" in text or "resistance" in text:
            nums = {k: float(v) for k, v in re.findall(r"\b([VIR])\s*=\s*([0-9]*\.?[0-9]+)", text)}
            res = self.physics.solve_ohms_law(nums)
            if len(res.get("steps", [])) == 0:
                return self._clarify(["Provide two of V, I, R (e.g., V=10, I=2)"])
            out = "; ".join(res.get("steps", []))
            return self._wrap(out, confidence=0.8, subject="ohms law")
        if "velocity" in text or "acceleration" in text or "u=" in text or "a=" in text or "t=" in text:
            nums = {k: float(v) for k, v in re.findall(r"\b([suvat])\s*=\s*([0-9]*\.?[0-9]+)", text)}
            res = self.physics.solve_motion(nums)
            if not res.get("steps"):
                return self._clarify(["Give u=, a=, t= (and I’ll compute v and s)"])
            return self._wrap(" | ".join(res["steps"]), confidence=0.75, subject="motion")
        # KB-backed explanation
        top = self.kb.search(text, tags=["physics"], top_k=1)
        if top and top[0][0] > 0.55:
            e = top[0][1]
            return self._wrap(f"{e.title}: {e.content}", confidence=top[0][0], subject=e.title, sources=[e.title])
        return self._clarify(["Ohm’s law", "SUVAT motion", "Physics definition from KB"])

    def _chem_route(self, text: str) -> Reply:
        m = re.search(r"molar mass of ([A-Za-z0-9]+)", text)
        if m:
            f = m.group(1)
            mm = self.chem.molar_mass(f)
            return self._wrap(f"Molar mass of {f} ≈ {mm:.3f} g/mol", confidence=0.85, subject=f"molar mass {f}")
        # element lookup
        m2 = re.search(r"(element|periodic)\s+([A-Z][a-z]?)", text)
        if m2:
            sym = m2.group(2)
            if sym in self.chem.table:
                el = self.chem.table[sym]
                return self._wrap(f"{sym}: {el['name']}, atomic mass {el['atomic_mass']}", confidence=0.8, subject=f"element {sym}")
        # KB-backed explanation
        top = self.kb.search(text, tags=["chemistry"], top_k=1)
        if top and top[0][0] > 0.55:
            e = top[0][1]
            return self._wrap(f"{e.title}: {e.content}", confidence=top[0][0], subject=e.title, sources=[e.title])
        return self._clarify(["Molar mass", "Element info", "Chemistry definition from KB"])

    def _quiz_route(self, text: str) -> Reply:
        m = re.search(r"(quiz|test|practice)\s+(on|about)\s+([a-z0-9\s\-]+)", text)
        topic = m.group(3).strip() if m else None
        q = self.quiz.generate(topic=topic)
        return self._wrap(f"Q: {q['question']}\n(Answer ready — say 'show answer')", confidence=0.7, subject=f"quiz:{topic or 'general'}")

    def _flash_route(self, text: str) -> Reply:
        if "add" in text and ":" in text:
            # format: flashcard add: front -> back
            m = re.search(r"add:\s*(.+?)\s*->\s*(.+)$", text)
            if m:
                front, back = m.group(1).strip(), m.group(2).strip()
                self.flash.add(front, back)
                return self._wrap(f"Added flashcard: {front}", confidence=0.8, subject=f"flash:{front}")
        if "review" in text or "queue" in text:
            q = self.flash.review_queue()
            if not q:
                return self._wrap("No flashcards due. Add some with: flashcard add: term -> definition", confidence=0.6)
            lines = []
            for cid, c in q:
                lines.append(f"- {c['front']} (id: {cid})")
            return self._wrap("Due for review:\n" + "\n".join(lines) + "\nRate with: grade <id> <0-5>", confidence=0.7)
        m2 = re.search(r"grade\s+([a-z0-9\-]+)\s+([0-5])", text)
        if m2:
            cid, q = m2.group(1), int(m2.group(2))
            if cid in self.flash.cards:
                self.flash.update(cid, q)
                return self._wrap(f"Recorded grade {q} for {cid}.", confidence=0.8)
        return self._wrap("Flashcards: use 'flashcard add: front -> back', 'flashcards review', or 'grade <id> <0-5>'.", confidence=0.6)

    def _kb_route(self, text: str, subject_hint: Optional[str]) -> Reply:
        # Learn: "learn: Title | Content | tags: a,b"
        if text.startswith("learn:"):
            # parse
            body = text[len("learn:"):].strip()
            parts = [p.strip() for p in body.split("|")]
            if len(parts) >= 2:
                title, content = parts[0], parts[1]
                tags = []
                if len(parts) >= 3 and parts[2].lower().startswith("tags:"):
                    tags = [t.strip() for t in parts[2][5:].split(",") if t.strip()]
                e = self.kb.add(title, content, tags, source="user")
                return self._wrap(f"Learned '{e.title}' (id: {e.id})", confidence=0.9, subject=e.title)
            return self._wrap("Format: learn: Title | Content | tags: t1,t2", confidence=0.5)
        if text.startswith("forget "):
            eid = text.split(" ", 1)[1].strip()
            ok = self.kb.forget(eid)
            return self._wrap("Forgot." if ok else "Entry not found.", confidence=0.7)
        if text.startswith("kb search"):
            q = text.split("kb search", 1)[1].strip() or subject_hint or ""
            results = self.kb.search(q, top_k=5)
            if not results:
                return self._wrap("No KB results.", confidence=0.6)
            lines = [f"{score:.2f} - {e.title} (id: {e.id})" for score, e in results]
            return self._wrap("KB results:\n" + "\n".join(lines), confidence=0.75)
        # General user question: try KB retrieval
        results = self.kb.search(text, top_k=3)
        if results and results[0][0] > 0.6:
            score, e = results[0]
            return self._wrap(f"{e.title}: {e.content}", confidence=score, subject=e.title, sources=[e.title])
        # Low confidence -> clarify
        return self._clarify(["Add to KB (use 'learn:')", "Search KB (use 'kb search <query>')", "Ask to define or explain"])

    def _smalltalk(self, text: str) -> Reply:
        greetings = ["hello", "hi", "hey"]
        thanks = ["thanks", "thank you", "cheers"]
        t = text.lower()
        if any(g in t for g in greetings):
            return self._wrap("Hey! What shall we explore today — math, physics, chemistry, or your KB?", confidence=0.6)
        if any(x in t for x in thanks):
            return self._wrap("You’re welcome. Want a random STEM fact?", confidence=0.6)
        # Fallback: attempt KB anyway
        results = self.kb.search(text, top_k=1)
        if results and results[0][0] > 0.55:
            score, e = results[0]
            return self._wrap(f"{e.title}: {e.content}", confidence=score, subject=e.title, sources=[e.title])
        return self._wrap("I can calculate, convert units, explain concepts, quiz you, and learn new facts. Try: convert 5 km to m; or learn: Title | Content | tags: ...", confidence=0.4)

# -------------------------------
# Profiles, Sessions, Transcripts
# -------------------------------
class ProfileManager:
    def __init__(self, store: JSONStore):
        self.store = store
        self.profiles = store.get("profiles") or {}

    def get_or_create(self, user_id: str, name="User", tz="Africa/Kampala", tone="teacher") -> UserProfile:
        if user_id in self.profiles:
            p = self.profiles[user_id]
        else:
            p = asdict(UserProfile(user_id=user_id, name=name, timezone=tz, tone=tone))
            self.profiles[user_id] = p
            self.store.set("profiles", self.profiles)
        return UserProfile(**p)

    def set_tone(self, user_id: str, tone: str):
        if user_id in self.profiles:
            self.profiles[user_id]["tone"] = tone
            self.store.set("profiles", self.profiles)

class SessionManager:
    def __init__(self, store: JSONStore):
        self.store = store
        self.sessions = store.get("sessions") or {}

    def start(self, user_id: str) -> Session:
        sid = slugify(f"{user_id}-{int(time.time())}")
        s = Session(session_id=sid, user_id=user_id)
        self.sessions[sid] = asdict(s)
        self.store.set("sessions", self.sessions)
        return s

    def save(self, session: Session):
        self.sessions[session.session_id] = asdict(session)
        self.store.set("sessions", self.sessions)

class Transcript:
    def __init__(self, store: JSONStore):
        self.store = store
        self.data = store.get("transcripts") or {}

    def append(self, session_id: str, role: str, text: str):
        if session_id not in self.data:
            self.data[session_id] = []
        self.data[session_id].append({
            "time": time.time(), "role": role, "text": text
        })
        self.store.set("transcripts", self.data)

    def export_md(self, session_id: str, path: str):
        entries = self.data.get(session_id, [])
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript {session_id}\n\n")
            for e in entries:
                ts = dt.datetime.fromtimestamp(e["time"]).isoformat(timespec="seconds")
                f.write(f"- [{ts}] {e['role']}: {e['text']}\n")

# -------------------------------
# CLI
# -------------------------------
HELP = """Commands you can try:
- convert 5 km to m
- calculate 2*(3+4)^2
- what is photosynthesis?
- physics: V=I*R with I=2, R=5
- chemistry: molar mass of H2O
- code: explain `for i in range(3): print(i)`
- quiz on photosynthesis
- flashcard add: Ohm's Law -> V = I * R
- flashcards review
- grade <card-id> <0-5>
- learn: Photosynthesis | Plants convert light to chemical energy... | tags: biology
- kb search photosynthesis
- forget <entry_id>
- tone teacher|peer|quizmaster
- export transcript
- exit
"""

def main():
    store = JSONStore()
    profiles = ProfileManager(store)
    user_id = "martin"
    profile = profiles.get_or_create(user_id=user_id, name="Martin", tz="Africa/Kampala", tone="teacher")
    kb = KB(store)
    # Seed example KB if empty
    if not kb.entries:
        kb.add("Ohm's Law", "Voltage equals current times resistance (V = I * R).", ["physics", "electricity", "uganda o level"], "nova_manual")
        kb.add("Photosynthesis", "Plants convert light energy into chemical energy (glucose) using CO2 and water, releasing oxygen.", ["biology", "uganda o level"], "nova_manual")
        kb.add("Density", "Density is mass per unit volume, ρ = m / V.", ["physics", "uganda o level"], "nova_manual")

    router = Router(store, profile, kb)
    sessions = SessionManager(store)
    transcripts = Transcript(store)
    session = sessions.start(user_id=user_id)

    print(f"Nova ready. Timezone: {profile.timezone}. Tone: {profile.tone}.")
    print(HELP)
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if text.lower().startswith("tone "):
            tone = text.split(" ", 1)[1].strip().lower()
            if tone in {"teacher", "peer", "quizmaster"}:
                profiles.set_tone(user_id, tone)
                router.dialogue.profile.tone = tone
                print(f"Tone set to {tone}.")
                continue
            else:
                print("Tone must be teacher|peer|quizmaster.")
                continue
        if text.lower() == "export transcript":
            path = os.path.join(DATA_DIR, f"transcript-{session.session_id}.md")
            transcripts.export_md(session.session_id, path)
            print(f"Transcript saved to {path}")
            continue

        transcripts.append(session.session_id, "user", text)
        reply = router.handle(text, session)
        print(f"Nova: {reply.text}")
        if reply.follow_up:
            print(f"(Follow-up) {reply.follow_up}")
        transcripts.append(session.session_id, "nova", reply.text)
        sessions.save(session)

if __name__ == "__main__":
    main()
