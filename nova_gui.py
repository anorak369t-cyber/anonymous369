import os
import json
import time
import math
import re
import tkinter as tk
from tkinter import scrolledtext, messagebox
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
try:
    import sympy
except ImportError:
    sympy = None

# Optional: spaCy for better NLU
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

DATA_DIR = os.path.join(os.path.dirname(__file__), "nova_data")
os.makedirs(DATA_DIR, exist_ok=True)

# --- Storage ---
class JSONStore:
    def __init__(self, fname):
        self.path = os.path.join(DATA_DIR, fname)
        self.data = self._load()
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    def save(self):
        with open(self.path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    def get(self):
        return self.data
    def set(self, val):
        self.data = val
        self.save()

# --- Data Models ---
@dataclass
class KBEntry:
    id: str
    title: str
    content: str
    tags: List[str] = field(default_factory=list)
    updated_at: float = field(default_factory=lambda: time.time())

# --- Knowledge Base ---
class KB:
    def __init__(self):
        self.store = JSONStore("kb.json")
        self.entries: Dict[str, KBEntry] = {}
        self._load()
    def _load(self):
        raw = self.store.get()
        if isinstance(raw, dict):
            for eid, e in raw.items():
                self.entries[eid] = KBEntry(**e)
        self._build_index()
    def _build_index(self):
        self.inv_index = {}
        for eid, e in self.entries.items():
            tokens = set(e.title.lower().split() + e.content.lower().split())
            for t in tokens:
                self.inv_index.setdefault(t, set()).add(eid)
    def add(self, title, content, tags):
        eid = re.sub(r'[^a-z0-9]+', '-', title.lower()) + "-" + str(int(time.time()))
        entry = KBEntry(id=eid, title=title, content=content, tags=tags)
        self.entries[eid] = entry
        self.store.set({k: asdict(v) for k, v in self.entries.items()})
        self._build_index()
        return entry
    def search(self, query, top_k=3):
        tokens = set(query.lower().split())
        found = set()
        for t in tokens:
            found |= self.inv_index.get(t, set())
        results = [(self.entries[eid].title, self.entries[eid].content) for eid in found]
        return results[:top_k] if results else []

# --- NLU ---
class NLU:
    def __init__(self):
        self.keywords = {
            "math": ["calculate", "solve", "math", "+", "-", "*", "/", "^"],
            "convert": ["convert", "to", "in", "km", "m", "kg", "g", "c", "f"],
            "define": ["define", "what is", "explain", "meaning"],
            "kb": ["learn:", "remember", "kb", "search kb", "forget"],
        }
    def analyze(self, text):
        text_l = text.lower()
        if nlp:
            doc = nlp(text_l)
            if any(t.pos_ == "NUM" for t in doc):
                return "math"
            # Demo: could add more advanced patterns here
        for intent, kws in self.keywords.items():
            if any(kw in text_l for kw in kws):
                return intent
        return "kb"

# --- Math ---
class MathSolver:
    def eval_expr(self, expr):
        if sympy:
            try:
                return str(sympy.sympify(expr))
            except Exception as e:
                return f"Error: {e}"
        else:
            # Fallback, insecure! Only for basic demo
            try:
                return str(eval(expr, {"__builtins__": None, "math": math}))
            except Exception as e:
                return f"Error: {e}"

# --- Unit Conversion ---
class UnitConverter:
    LENGTH = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0}
    MASS = {"mg": 0.001, "g": 1.0, "kg": 1000.0}
    def convert(self, val, from_u, to_u):
        fu, tu = from_u.lower(), to_u.lower()
        if fu in self.LENGTH and tu in self.LENGTH:
            return f"{val} {from_u} = {val*self.LENGTH[fu]/self.LENGTH[tu]:g} {to_u}"
        if fu in self.MASS and tu in self.MASS:
            return f"{val} {from_u} = {val*self.MASS[fu]/self.MASS[tu]:g} {to_u}"
        return "Unsupported units."

# --- Dictionary (Demo only) ---
class Dictionary:
    def __init__(self):
        self.data = {
            "photosynthesis": "Process by which plants use light to convert CO2 and water into glucose and oxygen.",
            "atom": "Smallest unit of matter that forms a chemical element.",
        }
    def define(self, term):
        t = term.lower().strip()
        return self.data.get(t, f"No definition for '{t}'.")

# --- Main Assistant Logic ---
class Nova:
    def __init__(self):
        self.kb = KB()
        self.nlu = NLU()
        self.math = MathSolver()
        self.units = UnitConverter()
        self.dict = Dictionary()
    def handle(self, text):
        intent = self.nlu.analyze(text)
        if intent == "math":
            # Try to extract math expr
            expr = re.findall(r"([0-9\+\-\*/\^\(\)\. ]+)", text)
            expr = expr[0] if expr else text
            return self.math.eval_expr(expr)
        elif intent == "convert":
            m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s+(?:to|in)\s+([a-zA-Z]+)", text)
            if m:
                val, from_u, to_u = float(m.group(1)), m.group(2), m.group(3)
                return self.units.convert(val, from_u, to_u)
            else:
                return "Please specify: e.g. '5 km to m'"
        elif intent == "define":
            m = re.search(r"(?:define|what is|explain|meaning of)\s+([a-z0-9\s\-]+)", text)
            term = m.group(1) if m else text
            return self.dict.define(term)
        elif intent == "kb":
            if text.lower().startswith("learn:"):
                parts = text[6:].split("|")
                if len(parts) >= 2:
                    title, content = parts[0].strip(), parts[1].strip()
                    tags = parts[2].strip().replace("tags:","").split(",") if len(parts)>2 else []
                    self.kb.add(title, content, tags)
                    return f"Learned: {title}"
                else:
                    return "Format: learn: Title | Content | tags:tag1,tag2"
            elif text.lower().startswith("search kb"):
                q = text[9:].strip()
                results = self.kb.search(q)
                return "\n".join([f"{i+1}. {t}: {c[:60]}" for i, (t, c) in enumerate(results)]) or "No results."
            else:
                # Try search by default
                results = self.kb.search(text)
                if results:
                    return "\n".join([f"{t}: {c[:60]}" for t, c in results])
                return "Ask me to learn or search the KB!"
        return "Sorry, I didn't understand. Try: 'convert 5 km to m', 'define atom', 'calculate 2*(3+4)^2', 'learn: Title | Content'"

# --- GUI ---
class NovaGUI:
    def __init__(self, root):
        self.nova = Nova()
        root.title("Nova STEM Assistant")
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, height=20, width=80, state='disabled')
        self.text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.entry_var = tk.StringVar()
        self.entry = tk.Entry(self.frame, textvariable=self.entry_var, width=80)
        self.entry.pack(padx=10, pady=(0,10), fill=tk.X)
        self.entry.bind("<Return>", self.on_send)
        self.print_message("Nova ready! Try: 'convert 5 km to m', 'define atom', 'calculate 2*(3+4)^2', 'learn: Title | Content'")

    def print_message(self, msg, sender="Nova"):
        self.text.config(state='normal')
        self.text.insert(tk.END, f"{sender}: {msg}\n")
        self.text.see(tk.END)
        self.text.config(state='disabled')

    def on_send(self, event=None):
        user_msg = self.entry_var.get().strip()
        if not user_msg: return
        self.print_message(user_msg, sender="You")
        try:
            reply = self.nova.handle(user_msg)
        except Exception as ex:
            reply = f"Error: {ex}"
        self.print_message(reply)
        self.entry_var.set("")

if __name__ == "__main__":
    root = tk.Tk()
    app = NovaGUI(root)
    root.mainloop()
