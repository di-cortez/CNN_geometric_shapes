import tkinter as tk
from ui_layout import Application

if __name__ == "__main__":
    """
    Ponto de entrada principal da aplicação.
    Cria a janela raiz, instancia a classe principal da UI e inicia o loop de eventos.
    """
    root = tk.Tk()
    
    # Cria a nossa aplicação, que é um Frame, e a coloca na janela
    app = Application(root, padding="10")
    app.pack(fill="both", expand=True)

    # Inicia o programa
    root.mainloop()