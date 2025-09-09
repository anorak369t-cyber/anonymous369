# nova_gui.py
import tkinter as tk
from tkinter import scrolledtext
import re, json
from datetime import datetime
from pathlib import Path

# ---------- Config ----------
NOVA_NAME = "Nova"
MEM_PATH = Path("nova_memory.json")

# ---------- Memory ----------
def load_memory():
    if MEM_PATH.exists():
        try: return json.loads(MEM_PATH.read_text(encoding="utf-8"))
        except: return {}
    return {}

def save_memory(mem): MEM_PATH.write_text(json.dumps(mem, indent=2), encoding="utf-8")
MEMORY = load_memory()

def remember(key, value): MEMORY[key] = value; save_memory(MEMORY)
def recall(key, default=None): return MEMORY.get(key, default)

# ---------- Skills ----------
def skill_greet(msg):
    m = re.search(r"\b(?:i am|i'm|my name is)\s+([A-Za-z\-']+)", msg, re.I)
    if m: remember("user_name", m.group(1).title())
    name = recall("user_name", "there")
    return f"Hey {name}, I’m {NOVA_NAME}. Ready to build, study, or create something sharp?"

def skill_time(_msg):
    now = datetime.now()
    return f"It’s {now.strftime('%A, %d %B %Y, %H:%M')}."

def skill_remember(msg):
    m = re.search(r"remember\s+([a-z0-9_\-\s]+?)\s*:\s*(.+)", msg, re.I)
    if not m: return "Use: remember <key>: <value>"
    key, val = m.group(1).strip().lower(), m.group(2).strip()
    remember(f"fact::{key}", val)
    return f"Got it. I’ll remember {key} as: {val}"

def skill_recall(msg):
    m = re.search(r"recall\s+([a-z0-9_\-\s]+)", msg, re.I)
    if not m:
        keys = [k.replace("fact::", "") for k in MEMORY if k.startswith("fact::")]
        return "Saved keys: " + ", ".join(keys) if keys else "No saved facts yet."
    key = m.group(1).strip().lower()
    val = recall(f"fact::{key}")
    return f"{key}: {val}" if val else f"Nothing saved for '{key}'."

def skill_calc(msg):
    m = re.search(r"(?:calc|calculate|compute)\s+(.+)", msg, re.I)
    if not m: return "Use: calc <expression> (e.g., calc (2+3)*4)"
    expr = m.group(1)
    if not re.fullmatch(r"[0-9\.\+\-\*\/\(\)\s]+", expr): return "Only numbers and + - * / allowed."
    try: return f"{expr} = {eval(expr, {'__builtins__': {}})}"
    except: return "That didn’t compute. Try a simpler expression."

def skill_study(msg):
    m = re.search(r"(?:study|outline|explain)\s+(.+)", msg, re.I)
    if not m: return "Use: study <topic> (e.g., study Newton’s Laws)"
    topic = m.group(1).strip().title()
    return (
        f"{topic} — PCM Study Outline:\n"
        "1) Definition: Clear, concise, with units.\n"
        "2) Core formulas: Variables + conditions.\n"
        "3) Common traps: Units, signs, approximations.\n"
        "4) Worked example: Step-by-step with logic.\n"
        "5) Quick checks: Dimensional sanity, edge cases."
    )

def skill_tagline(msg):
    m = re.search(r"(?:tagline|brand|slogan)\s+(.+)", msg, re.I)
    if not m: return "Use: tagline <brand name> (e.g., tagline Creana)"
    brand = m.group(1).strip().title()
    return (
        f"{brand} — Sample Taglines:\n"
        f"- \"{brand}: Where ideas find form.\"\n"
        f"- \"{brand}: Crafted for clarity, built for impact.\"\n"
        f"- \"{brand}: Your story, beautifully told.\""
    )

def skill_help(_msg):
    return (
        f"I’m {NOVA_NAME}. I can:\n"
        "- greet → Learn your name\n"
        "- time → Show current time\n"
        "- remember <key>: <value> → Save a fact\n"
        "- recall <key> → Retrieve a fact\n"
        "- calc <expr> → Compute math\n"
        "- study <topic> → PCM outline\n"
        "- tagline <brand> → Generate slogans"
    )

# ---------- Router ----------
INTENTS = [
    (r"\b(hi|hello|hey|kopango|nango)\b", skill_greet),
    (r"\b(time|what.*time)\b", skill_time),
    (r"\bremember\b", skill_remember),
    (r"\brecall\b", skill_recall),
    (r"\b(calc|calculate|compute)\b", skill_calc),
    (r"\b(study|outline|explain)\b", skill_study),
    (r"\b(tagline|brand|slogan)\b", skill_tagline),
    (r"\b(help|what can you do)\b", skill_help),
]

def respond(msg: str) -> str:
    msg = msg.strip()
    if msg.lower() in {"who are you", "who are u"}:
        return f"I’m {NOVA_NAME} — curious, clear, and built to serve with precision."
    for pattern, handler in INTENTS:
        if re.search(pattern, msg, re.I): return handler(msg)
    name = recall("user_name", "")
    return f"{name}, tell me what you’re solving or building. I’ll help map the fastest path."

# ---------- GUI ----------
class NovaGUI:
    def __init__(self, root):
        root.title(f"{NOVA_NAME} Assistant")
        root.geometry("600x500")
        root.configure(bg="#f0f0f0")

        self.chat = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Segoe UI", 11))
        self.chat.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat.insert(tk.END, f"{NOVA_NAME}: Hello! Type 'help' to see what I can do.\n")
        self.chat.config(state=tk.DISABLED)

        self.entry = tk.Entry(root, font=("Segoe UI", 11))
        self.entry.pack(padx=10, pady=(0,10), fill=tk.X)
        self.entry.bind("<Return>", self.send)

        self.send_btn = tk.Button(root, text="Send", command=self.send, font=("Segoe UI", 10))
        self.send_btn.pack(pady=(0,10))

    def send(self, event=None):
        user_msg = self.entry.get().strip()
        if not user_msg: return
        self.entry.delete(0, tk.END)
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, f"You: {user_msg}\n")
        reply = respond(user_msg)
        self.chat.insert(tk.END, f"{NOVA_NAME}: {reply}\n\n")
        self.chat.config(state=tk.DISABLED)
        self.chat.yview(tk.END)

# ---------- Run ----------
if __name__ == "__main__":
    root = tk.Tk()
    app = NovaGUI(root)
    root.mainloop()
