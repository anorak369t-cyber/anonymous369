import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
import json, os

KB_FILE = "kb.json"

# ===== KB Management =====
def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_kb(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

def import_kb():
    path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
    if path:
        with open(path, "r", encoding="utf-8") as f:
            KB.update(json.load(f))
        save_kb(KB)
        messagebox.showinfo("Import", "KB imported successfully.")

def export_kb():
    path = filedialog.asksaveasfilename(defaultextension=".json")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(KB, f, ensure_ascii=False, indent=2)
        messagebox.showinfo("Export", "KB exported successfully.")

# ===== Core KB =====
KB = {
    "hello": "Hi there! How can I help you today?",
    "what is your name": "I'm Nova, your Python chatbot."
}
KB.update(load_kb())

# ===== Chatbot Logic =====
class Chatbot:
    def __init__(self, teach_mode=True):
        self.teach_mode = teach_mode

    def respond(self, query):
        q = query.lower().strip()
        if q in KB:
            return KB[q]
        for key in KB:
            if key in q or q in key:
                return KB[key]
        return None

    def learn(self, query, answer):
        KB[query.lower().strip()] = answer.strip()
        save_kb(KB)

# ===== GUI Class =====
class NovaGUI:
    def __init__(self, root):
        self.bot = Chatbot()
        self.root = root
        self.root.title("Nova - STEM Chatbot")

        # Theme
        self.dark_mode = False
        self.font_size = 12

        # Chat log
        self.chat_log = tk.Text(root, state="disabled", wrap="word", bg="#f0f0f0", font=("Arial", self.font_size))
        self.chat_log.pack(padx=10, pady=10, fill="both", expand=True)

        # Entry
        self.entry = tk.Entry(root, font=("Arial", self.font_size))
        self.entry.pack(padx=10, pady=(0,10), fill="x")
        self.entry.bind("<Return>", self.send_message)

        # Menu
        self._build_menu()

    def _build_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        kb_menu = tk.Menu(menu, tearoff=0)
        kb_menu.add_command(label="View/Edit KB", command=self.view_kb)
        kb_menu.add_command(label="Import KB", command=import_kb)
        kb_menu.add_command(label="Export KB", command=export_kb)
        menu.add_cascade(label="Knowledge Base", menu=kb_menu)

        mode_menu = tk.Menu(menu, tearoff=0)
        mode_menu.add_command(label="Toggle Teach Mode", command=self.toggle_teach_mode)
        mode_menu.add_command(label="Flashcard Mode", command=self.flashcard_mode)
        menu.add_cascade(label="Modes", menu=mode_menu)

        view_menu = tk.Menu(menu, tearoff=0)
        view_menu.add_command(label="Toggle Dark Mode", command=self.toggle_dark_mode)
        view_menu.add_command(label="Increase Font", command=lambda: self.adjust_font(2))
        view_menu.add_command(label="Decrease Font", command=lambda: self.adjust_font(-2))
        menu.add_cascade(label="View", menu=view_menu)

    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input:
            return
        self.entry.delete(0, tk.END)
        self._append_chat(f"You: {user_input}")

        response = self.bot.respond(user_input)
        if response:
            self._append_chat(f"Nova: {response}")
        elif self.bot.teach_mode:
            answer = simpledialog.askstring("Teach Nova", f"I don't know '{user_input}'. What should I say?")
            if answer:
                self.bot.learn(user_input, answer)
                self._append_chat("Nova: Got it! I’ll remember that.")
            else:
                self._append_chat("Nova: Hmm, I’m not sure about that.")
        else:
            self._append_chat("Nova: I don’t know that one. Teach mode is off.")

    def _append_chat(self, message):
        self.chat_log.config(state="normal")
        self.chat_log.insert(tk.END, message + "\n")
        self.chat_log.config(state="disabled")
        self.chat_log.see(tk.END)

    def toggle_teach_mode(self):
        self.bot.teach_mode = not self.bot.teach_mode
        status = "ON" if self.bot.teach_mode else "OFF"
        messagebox.showinfo("Teach Mode", f"Teach mode is now {status}.")

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        bg = "#2e2e2e" if self.dark_mode else "#f0f0f0"
        fg = "#ffffff" if self.dark_mode else "#000000"
        self.chat_log.config(bg=bg, fg=fg)
        self.entry.config(bg=bg, fg=fg)

    def adjust_font(self, delta):
        self.font_size = max(8, self.font_size + delta)
        self.chat_log.config(font=("Arial", self.font_size))
        self.entry.config(font=("Arial", self.font_size))

    def view_kb(self):
        kb_win = tk.Toplevel(self.root)
        kb_win.title("Knowledge Base Editor")
        text = tk.Text(kb_win, wrap="word")
        text.pack(fill="both", expand=True)
        text.insert("1.0", json.dumps(KB, indent=2))
        def save_edits():
            try:
                edited = json.loads(text.get("1.0", tk.END))
                KB.clear()
                KB.update(edited)
                save_kb(KB)
                messagebox.showinfo("Saved", "KB updated.")
                kb_win.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid JSON: {e}")
        tk.Button(kb_win, text="Save", command=save_edits).pack(pady=5)

    def flashcard_mode(self):
        if not KB:
            messagebox.showinfo("Flashcards", "KB is empty.")
            return
        flash_win = tk.Toplevel(self.root)
        flash_win.title("Flashcard Mode")
        keys = list(KB.keys())
        index = [0]

        question = tk.Label(flash_win, text=keys[index[0]], font=("Arial", 14))
        question.pack(pady=10)
        answer = tk.Label(flash_win, text="", font=("Arial", 12))
        answer.pack(pady=10)

        def show_answer():
            answer.config(text=KB[keys[index[0]]])

        def next_card():
            index[0] = (index[0] + 1) % len(keys)
            question.config(text=keys[index[0]])
            answer.config(text="")

        tk.Button(flash_win, text="Show Answer", command=show_answer).pack()
        tk.Button(flash_win, text="Next", command=next_card).pack()

# ===== Run App =====
if __name__ == "__main__":
    root = tk.Tk()
    app = NovaGUI(root)
    root.mainloop()
