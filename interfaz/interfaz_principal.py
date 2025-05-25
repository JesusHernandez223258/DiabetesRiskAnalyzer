import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import pandas as pd

from datos.generador_datos import generar_datos_ejemplo
from visualizacion.graficos import crear_graficos

class DiabetesRiskAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Factores de Riesgo - Diabetes Tipo 2")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        self.df = None
        self.categorical_vars = []
        self.continuous_vars = []

        self.setup_style()
        self.create_interface()

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0', foreground='#34495e')
        style.configure('Modern.TButton', font=('Arial', 10), padding=(10, 5))
        style.map('Modern.TButton', background=[('active', '#3498db'), ('!active', '#2980b9')])

    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_label = ttk.Label(main_frame, 
            text="🏥 Análisis de Factores de Riesgo - Diabetes Tipo 2",
            style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        left_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        ttk.Label(left_frame, text="Gestión de Datos", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        ttk.Button(left_frame, text="📁 Cargar Dataset", 
            command=self.load_data, style='Modern.TButton').grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="🎲 Generar Datos de Ejemplo", 
            command=self.generate_sample_data, style='Modern.TButton').grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="🧹 Limpiar Datos", 
            command=self.clean_data, style='Modern.TButton').grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(left_frame, text="Variables Categóricas", style='Header.TLabel').grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        self.cat_listbox = tk.Listbox(left_frame, height=4, selectmode=tk.MULTIPLE)
        self.cat_listbox.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Label(left_frame, text="Variables Continuas", style='Header.TLabel').grid(row=7, column=0, sticky=tk.W, pady=(0, 5))
        self.cont_listbox = tk.Listbox(left_frame, height=4, selectmode=tk.MULTIPLE)
        self.cont_listbox.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(left_frame, text="Análisis y Visualización", style='Header.TLabel').grid(row=10, column=0, sticky=tk.W, pady=(0, 10))

        ttk.Button(left_frame, text="📊 Generar Gráficos de Barras", 
            command=self.create_bar_plots, style='Modern.TButton').grid(row=11, column=0, sticky=(tk.W, tk.E), pady=2)

        center_frame = ttk.LabelFrame(main_frame, text="Información del Dataset", padding="10")
        center_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        self.info_text = scrolledtext.ScrolledText(center_frame, width=40, height=25)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        right_frame = ttk.LabelFrame(main_frame, text="Vista Previa de Gráficos", padding="10")
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def generate_sample_data(self):
        self.df = generar_datos_ejemplo()
        self.update_info()

    def load_data(self):
        messagebox.showinfo("Cargar Datos", "Funcionalidad aún no implementada.")

    def clean_data(self):
        messagebox.showinfo("Limpiar Datos", "Funcionalidad aún no implementada.")

    def create_bar_plots(self):
        crear_graficos(self.df, self.ax)
        self.canvas.draw()

    def update_info(self):
        self.info_text.delete("1.0", tk.END)
        if self.df is not None:
            self.info_text.insert(tk.END, str(self.df.describe()))
