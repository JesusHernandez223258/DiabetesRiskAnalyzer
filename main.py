import tkinter as tk
from interfaz.interfaz_principal import DiabetesRiskAnalyzer

if __name__ == '__main__':
    root = tk.Tk()
    app = DiabetesRiskAnalyzer(root)
    root.mainloop()
