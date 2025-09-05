import json
import os

KB_FILE = "kb.json"

def load_kb() -> Dict[str, str]:
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_kb(kb: Dict[str, str]):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# -----------------------------
# Simple in-memory knowledge base
# -----------------------------
KB: Dict[str, str] = {
    "what is newton's second law": "Newton's second law states that force equals mass times acceleration: F = m·a.",
    "define momentum": "Momentum is the product of mass and velocity (p = m·v). It’s a vector quantity.",
    "what is ohm's law": "Ohm's law relates voltage, current, and resistance: V = I·R.",
    "what is github pages": "GitHub Pages is a static site hosting service that takes HTML/CSS/JS from a repo and publishes it at a URL.",
    "how do i make quizzes online": "You can build quizzes with plain HTML forms + JS, or frameworks like React/Vue. For static hosting, bundle assets and deploy to GitHub Pages.",
}

# -----------------------------
# Safety filter (very simple)
# -----------------------------
DISALLOWED = [
    r"make (a )?bomb", r"how to hack", r"credit card generator", r"harm (someone|people|myself)",
    r"kill|suicide|poison", r"ddos", r"ransomware", r"exploit", r"malware"
]

def is_disallowed(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in DISALLOWED)

# -----------------------------
# Small utilities / tools
# -----------------------------
def calculator(expr: str) -> str:
    # Safe-ish eval: numbers, + - * / ( ) ^ **, sqrt, sin, cos, tan, log
    allowed = set("0123456789.+-*/() ^")
    expr = expr.replace("^", "**")
    if not set(expr) <= allowed and not re.search(r"(sqrt|sin|cos|tan|log)", expr):
        return "I only support basic arithmetic and math functions: + - * / ^ sqrt sin cos tan log."
    env = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log
    }
    try:
        val = eval(expr, {"__builtins__": {}}, env)
        return f"{val}"
    except Exception as e:
        return f"I couldn't compute that. Try something like: 2^3 + sqrt(16)."

UNIT_MAP = {
    # length (meters)
    ("cm", "m"): 0.01, ("m", "cm"): 100.0,
    ("km", "m"): 1000.0, ("m", "km"): 0.001,
    # mass (kilograms)
    ("g", "kg"): 0.001, ("kg", "g"): 1000.0,
    # time (seconds)
    ("min", "s"): 60.0, ("s", "min"): 1/60.0,
    ("hr", "s"): 3600.0, ("s", "hr"): 1/3600.0,
}

def convert_units(amount: float, src: str, dst: str) -> Optional[float]:
    key = (src.lower(), dst.lower())
    factor = UNIT_MAP.get(key)
    if factor is None:
        return None
    return amount * factor

def now_local() -> str:
    # Local time of the machine running the script
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# Simple memory to keep context
# -----------------------------
class Memory:
    def __init__(self, size: int = 6):
        self.turns: List[Tuple[str, str]] = []
        self.user_name: Optional[str] = None
        self.size = size

    def remember(self, user: str, bot: str):
        self.turns.append((user, bot))
        if len(self.turns) > self.size:
            self.turns.pop(0)

    def set_name(self, name: str):
        self.user_name = name.strip().title()

    def last_user_message(self) -> Optional[str]:
        return self.turns[-1][0] if self.turns else None

# -----------------------------
# Intent patterns
# -----------------------------
INTENTS = [
    ("greet", r"^(hi|hello|hey|yo|sup)\b"),
    ("set_name", r"my name is ([a-zA-Z][a-zA-Z\s'-]{1,30})"),
    ("ask_time", r"\b(time|date)\b"),
    ("calc", r"(?:calc|calculate|compute|evaluate)\s*:(.+)$"),
    ("calc_inline", r"^([0-9\(\)\+\-\*\/\.\s\^]+)$"),
    ("convert", r"convert\s*([0-9\.]+)\s*([a-zA-Z]+)\s*to\s*([a-zA-Z]+)"),
    ("define", r"^(what is|define)\s+(.+)$"),
    ("thanks", r"\b(thanks|thank you|appreciate it)\b"),
    ("bye", r"\b(bye|goodbye|see ya|later)\b"),
]

SMALLTALK = [
    "Tell me more about that.",
    "Interesting—what’s the goal behind it?",
    "What’s the toughest part for you right now?",
    "If it worked perfectly, what would that look like?",
]

# -----------------------------
# Core NLU / routing
# -----------------------------
class Chatbot:
    def __init__(self, name: str = "Nova"):
        self.name = name
        self.mem = Memory()

    def respond(self, text: str) -> str:
        if not text or not text.strip():
            return "Say anything—math, conversions, definitions, or ask me about physics and coding."
        if is_disallowed(text):
            return "I can’t help with that. Try a different topic—learning, tools, or problem‑solving."

        t = text.strip()
        low = t.lower()

        # Intent detection
        for intent, pattern in INTENTS:
            m = re.search(pattern, low)
            if not m:
                continue
            if intent == "greet":
                return self._greet()
            if intent == "set_name":
                self.mem.set_name(m.group(1))
                return f"Nice to meet you, {self.mem.user_name}."
            if intent == "ask_time":
                return f"It’s {now_local()}."
            if intent == "calc":
                expr = m.group(1).strip()
                return f"Result: {calculator(expr)}"
            if intent == "calc_inline":
                expr = m.group(1).strip()
                # Only treat as calc if it actually looks like math, not a random number sentence
                if re.search(r"[+\-*/^()]", expr):
                    return f"Result: {calculator(expr)}"
            if intent == "convert":
                amount = float(m.group(1))
                src = m.group(2)
                dst = m.group(3)
                val = convert_units(amount, src, dst)
                return f"{amount} {src} = {val} {dst}" if val is not None else "I don’t know that unit conversion yet."
            if intent == "define":
                query = m.group(2).strip(" ?.")
                ans = self._kb_lookup(query)
                return ans

            if intent == "thanks":
                return "You’re welcome!"
            if intent == "bye":
                return "Goodbye! Keep building."

        # Fallbacks: KB search, smalltalk, or prompt for clarification
        kb_ans = self._kb_lookup(low, loose=True)
        if kb_ans:
            return kb_ans

        # Follow-up referencing prior message
        prev = self.mem.last_user_message()
        if prev and len(t.split()) <= 4:
            return f"Are you referring to: “{prev}”? Tell me what you want to do with it."

        return self._smalltalk()

    def _greet(self) -> str:
        if self.mem.user_name:
            return f"Hey {self.mem.user_name}! What are we building today?"
        return "Hey! I’m Nova. What should we explore—math, physics, or code?"

    def _kb_lookup(self, query: str, loose: bool = False) -> Optional[str]:
        q = query.lower().strip()
        if not loose:
            return KB.get(q)
        # Loose contains-based match
        best_key = None
        for k in KB:
            if all(tok in q for tok in k.split()):
                best_key = k
                break
        return KB.get(best_key) if best_key else None

    def _smalltalk(self) -> str:
        # Cycle a bit using time entropy
        idx = int(time.time()) % len(SMALLTALK)
        return SMALLTALK[idx]

# -----------------------------
# Command-line loop
# -----------------------------
def main():
    bot = Chatbot()
    print("Nova (Python Chatbot). Type 'exit' or 'quit' to leave.")
    print("Examples:")
    print("  - hi")
    print("  - my name is Martin")
    print("  - calculate: 2^5 + sqrt(49)")
    print("  - 3*(2+5)")
    print("  - convert 120 cm to m")
    print("  - what is newton's second law")
    print("  - time")
    print("-" * 48)
    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if user.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        reply = bot.respond(user)
        # remember context
        bot.mem.remember(user, reply)
        print(reply)

if __name__ == "__main__":
    main()
