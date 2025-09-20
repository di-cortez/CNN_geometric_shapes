import tkinter as tk
from gui.views.main_window import Application

if __name__ == "__main__":
    """Launch the Neural Network Analyzer GUI."""
    root = tk.Tk()
    app = Application(root, padding="10")
    app.pack(fill="both", expand=True)
    root.mainloop()
