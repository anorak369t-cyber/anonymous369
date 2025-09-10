import json
import os
from typing import Dict

# ===== Persistent KB storage =====
KB_FILE = "kb.json"

def load_kb() -> Dict[str, str]:
    """Load saved KB from file if it exists."""
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_kb(kb: Dict[str, str]):
    """Save KB to file."""
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

# ===== Built-in KB =====
KB: Dict[str, str] = {
    "hello": "Hi there! How can I help you today?",
    "how are you": "I'm just code, but I'm running smoothly!",
    "what is your name": "I'm Nova, your Python chatbot."
}

# ===== Chatbot Class =====
class Chatbot:
    def __init__(self):
        pass

    def respond(self, user_input: str) -> str:
        low = user_input.lower().strip()

        # 1. Check KB for exact match
        if low in KB:
            return KB[low]

        # 2. Loose match search
        kb_ans = self._kb_lookup(low, loose=True)
        if kb_ans:
            return kb_ans

        # 3. Ask user to teach
        print("I don't know that yet. Can you teach me the answer?")
        new_answer = input("Your answer: ").strip()
        if new_answer:
            KB[low] = new_answer
            save_kb(KB)
            return "Got it! I’ll remember that."
        else:
            return self._smalltalk()

    def _kb_lookup(self, query: str, loose: bool = False) -> str:
        """Find KB entry by loose matching."""
        if loose:
            for key, val in KB.items():
                if key in query or query in key:
                    return val
        return ""

    def _smalltalk(self) -> str:
        """Fallback small talk."""
        return "Hmm, I’m not sure about that. Tell me more!"

# ===== Main Loop =====
def main():
    # Load saved KB and merge with built-in KB
    stored_kb = load_kb()
    KB.update(stored_kb)

    bot = Chatbot()
    print("Nova (Python Chatbot). Type 'exit' or 'quit' to leave.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Nova: Goodbye!")
            break
        response = bot.respond(user_input)
        print(f"Nova: {response}")

if __name__ == "__main__":
    main()