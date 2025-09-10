import tkinter as tk
from tkinter import messagebox, filedialog
import json, os

KB_FILE = "kb.json"

def load_kb():
    if os.path.exists(KB_FILE):
        with open(KB_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_kb(kb):
    with open(KB_FILE, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

class KBEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Nova KB Editor")
        self.kb = load_kb()

        self.listbox = tk.Listbox(root)
        self.listbox.pack(side="left", fill="y", padx=5, pady=5)
        self.listbox.bind("<<ListboxSelect>>", self.load_entry)

        self.editor = tk.Text(root, wrap="word")
        self.editor.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self._build_menu()
        self.refresh_list()

    def _build_menu(self):
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="Save", command=self.save_entry)
        file_menu.add_command(label="Add New", command=self.add_entry)
        file_menu.add_command(label="Delete", command=self.delete_entry)
        file_menu.add_command(label="Export KB", command=self.export_kb)
        file_menu.add_command(label="Import KB", command=self.import_kb)
        menu.add_cascade(label="File", menu=file_menu)

    def refresh_list(self):
        self.listbox.delete(0, tk.END)
        for key in self.kb:
            self.listbox.insert(tk.END, key)

    def load_entry(self, event=None):
        selection = self.listbox.curselection()
        if not selection:
            return
        key = self.listbox.get(selection[0])
        value = self.kb[key]
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", json.dumps({key: value}, indent=2))

    def save_entry(self):
        try:
            data = json.loads(self.editor.get("1.0", tk.END))
            for key, value in data.items():
                self.kb[key] = value
            save_kb(self.kb)
            self.refresh_list()
            messagebox.showinfo("Saved", "Entry saved successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid JSON: {e}")

    def add_entry(self):
        new_key = tk.simpledialog.askstring("New Entry", "Enter the question:")
        if new_key:
            self.kb[new_key] = "Type your answer here..."
            self.refresh_list()
            self.listbox.select_set(tk.END)
            self.load_entry()

    def delete_entry(self):
        selection = self.listbox.curselection()
        if not selection:
            return
        key = self.listbox.get(selection[0])
        if messagebox.askyesno("Delete", f"Delete entry '{key}'?"):
            del self.kb[key]
            save_kb(self.kb)
            self.refresh_list()
            self.editor.delete("1.0", tk.END)

    def export_kb(self):
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.kb, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Export", "KB exported successfully.")

    def import_kb(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if path:
            with open(path, "r", encoding="utf-8") as f:
                imported = json.load(f)
                self.kb.update(imported)
                save_kb(self.kb)
                self.refresh_list()
                messagebox.showinfo("Import", "KB imported successfully.")

# ===== Run Editor =====
if __name__ == "__main__":
    root = tk.Tk()
    app = KBEditor(root)
    root.mainloop()
