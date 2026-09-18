"""Virtual Band AI — Ponto de Entrada Principal (v0.1-A).

Executa a interface gráfica e o motor de áudio.
"""

import sys
import tkinter as tk
from app.ui.main_window import MainWindow


def main():
    """Inicializa a aplicação Virtual Band AI."""
    root = tk.Tk()
    app = MainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
