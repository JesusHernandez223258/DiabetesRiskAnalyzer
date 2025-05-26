import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # Importar NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io

from datos.generador_datos import generar_datos_ejemplo
import visualizacion.graficos as viz_graficos

class DiabetesRiskAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Datos Exploratorio")
        self.root.geometry("1500x900") # Un poco más de espacio para la barra de herramientas
        self.root.configure(bg='#f0f0f0')
        
        self.df = None
        self.original_df = None # Para guardar el df original antes de filtrar (si implementamos filtros)
        self.categorical_vars = []
        self.continuous_vars = []
        self.plot_generated = False

        self.setup_style()
        self.create_interface()

    def setup_style(self):
        # ... (sin cambios)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0', foreground='#34495e')
        style.configure('Modern.TButton', font=('Arial', 10), padding=(10, 5))


    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=2) # Controles (un poco más de peso)
        main_frame.columnconfigure(1, weight=3) # Info
        main_frame.columnconfigure(2, weight=5) # Gráfico (más peso para el gráfico y su toolbar)
        main_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="📊 Analizador de Datos Exploratorio", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=tk.W)

        # --- Panel Izquierdo: Controles ---
        # ... (contenido del panel izquierdo hasta los botones de generar y guardar sin cambios significativos) ...
        left_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)

        ttk.Label(left_frame, text="Gestión de Datos", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        ttk.Button(left_frame, text="📁 Cargar Dataset CSV", 
            command=self.load_data, style='Modern.TButton').grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(left_frame, text="🎲 Generar Datos de Ejemplo", 
            command=self.generate_sample_data, style='Modern.TButton').grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        ttk.Button(left_frame, text="🧹 Limpiar Datos (Quitar NaNs)", 
            command=self.clean_data, style='Modern.TButton').grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)

        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(left_frame, text="Variables Categóricas Detectadas", style='Header.TLabel').grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        self.cat_listbox = tk.Listbox(left_frame, height=4, exportselection=False) 
        self.cat_listbox.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Label(left_frame, text="Variables Continuas Detectadas", style='Header.TLabel').grid(row=7, column=0, sticky=tk.W, pady=(0, 5))
        self.cont_listbox = tk.Listbox(left_frame, height=4, exportselection=False)
        self.cont_listbox.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(left_frame, text="Configuración de Gráfico", style='Header.TLabel').grid(row=10, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(left_frame, text="Tipo de Gráfico:").grid(row=11, column=0, sticky=tk.W, pady=1)
        self.plot_type_var = tk.StringVar()
        self.plot_type_combo = ttk.Combobox(left_frame, textvariable=self.plot_type_var, 
                                            values=["Barra", "Caja (Box)", "Violín", "Histograma"], state="readonly")
        self.plot_type_combo.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=1)
        self.plot_type_combo.set("Barra")

        ttk.Label(left_frame, text="Variable X:").grid(row=13, column=0, sticky=tk.W, pady=1)
        self.x_var = tk.StringVar()
        self.x_var.trace_add("write", self.on_variable_selection_change)
        self.x_combo = ttk.Combobox(left_frame, textvariable=self.x_var, state="readonly")
        self.x_combo.grid(row=14, column=0, sticky=(tk.W, tk.E), pady=1)

        ttk.Label(left_frame, text="Variable Y (si aplica):").grid(row=15, column=0, sticky=tk.W, pady=1)
        self.y_var = tk.StringVar()
        self.y_combo = ttk.Combobox(left_frame, textvariable=self.y_var, state="readonly")
        self.y_combo.grid(row=16, column=0, sticky=(tk.W, tk.E), pady=1)

        ttk.Label(left_frame, text="Variable Hue (Agrupar por):").grid(row=17, column=0, sticky=tk.W, pady=1)
        self.hue_var = tk.StringVar()
        self.hue_combo = ttk.Combobox(left_frame, textvariable=self.hue_var, state="readonly")
        self.hue_combo.grid(row=18, column=0, sticky=(tk.W, tk.E), pady=1)

        ttk.Label(left_frame, text="Título del Gráfico (opcional):").grid(row=19, column=0, sticky=tk.W, pady=(5,0))
        self.plot_title_var = tk.StringVar()
        self.plot_title_entry = ttk.Entry(left_frame, textvariable=self.plot_title_var)
        self.plot_title_entry.grid(row=20, column=0, sticky=(tk.W, tk.E), pady=1)

        ttk.Label(left_frame, text="Etiqueta Eje X (opcional):").grid(row=21, column=0, sticky=tk.W, pady=1)
        self.plot_xlabel_var = tk.StringVar()
        self.plot_xlabel_entry = ttk.Entry(left_frame, textvariable=self.plot_xlabel_var)
        self.plot_xlabel_entry.grid(row=22, column=0, sticky=(tk.W, tk.E), pady=1)

        ttk.Label(left_frame, text="Etiqueta Eje Y (opcional):").grid(row=23, column=0, sticky=tk.W, pady=1)
        self.plot_ylabel_var = tk.StringVar()
        self.plot_ylabel_entry = ttk.Entry(left_frame, textvariable=self.plot_ylabel_var)
        self.plot_ylabel_entry.grid(row=24, column=0, sticky=(tk.W, tk.E), pady=1)
        
        self.generate_plot_button = ttk.Button(left_frame, text="📊 Generar Gráfico", 
            command=self.generate_selected_plot, style='Modern.TButton')
        self.generate_plot_button.grid(row=25, column=0, sticky=(tk.W, tk.E), pady=(10,2))

        self.save_plot_button = ttk.Button(left_frame, text="💾 Guardar Gráfico",
                                   command=self.save_plot, style='Modern.TButton')
        self.save_plot_button.grid(row=26, column=0, sticky=(tk.W, tk.E), pady=2)

        # --- Panel Central: Información del Dataset ---
        center_frame = ttk.LabelFrame(main_frame, text="Información del Dataset", padding="10")
        center_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        center_frame.rowconfigure(0, weight=1)
        center_frame.columnconfigure(0, weight=1)
        self.info_text = scrolledtext.ScrolledText(center_frame, width=50, height=25, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # --- Panel Derecho: Vista Previa de Gráficos ---
        right_frame = ttk.LabelFrame(main_frame, text="Vista Previa de Gráficos", padding="10")
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        right_frame.rowconfigure(0, weight=1) # Para la toolbar
        right_frame.rowconfigure(1, weight=10) # Para el canvas del gráfico (más peso)
        right_frame.columnconfigure(0, weight=1)
        
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        
        # AÑADIR LA BARRA DE HERRAMIENTAS DE NAVEGACIÓN
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky=tk.EW) # Colocarla arriba del canvas

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S)) # Canvas debajo de la toolbar

        self.update_ui_element_states()

    # ... (on_variable_selection_change, update_variable_lists_and_combos sin cambios mayores) ...
    def on_variable_selection_change(self, *args):
        """Callback cuando cambia una selección de variable, para actualizar estados de UI."""
        self.update_ui_element_states()

    def update_variable_lists_and_combos(self):
        self.cat_listbox.delete(0, tk.END)
        self.cont_listbox.delete(0, tk.END)
        
        self.categorical_vars = []
        self.continuous_vars = []

        # Restablecer DataFrame al original si se implementaron filtros
        if self.original_df is not None:
            self.df = self.original_df.copy()

        if self.df is not None:
            for col in self.df.columns:
                if self.df[col].dtype in ['object', 'string', 'bool', 'category']:
                    self.categorical_vars.append(col)
                elif pd.api.types.is_numeric_dtype(self.df[col]):
                    # Considerar una columna numérica como categórica si tiene pocos valores únicos
                    # Y el dataset es suficientemente grande para que no sea una casualidad
                    if self.df[col].nunique() < 20 and len(self.df) > 50: # Aumentado el umbral de nunique
                        # Y si esos valores únicos parecen discretos (ej. enteros)
                        if pd.api.types.is_integer_dtype(self.df[col]) or \
                           all(self.df[col].dropna().apply(lambda x: float(x).is_integer())):
                             self.categorical_vars.append(col)
                        else:
                            self.continuous_vars.append(col)
                    else:
                        self.continuous_vars.append(col)
                else: 
                    self.categorical_vars.append(col) 
            
            self.categorical_vars = sorted(list(set(self.categorical_vars)))
            self.continuous_vars = sorted(list(set(c for c in self.continuous_vars if c not in self.categorical_vars)))

            for var in self.categorical_vars:
                self.cat_listbox.insert(tk.END, var)
            for var in self.continuous_vars:
                self.cont_listbox.insert(tk.END, var)

        all_vars_for_x = [""] + sorted(list(set(self.categorical_vars + self.continuous_vars)))
        cat_vars_with_none = [""] + self.categorical_vars
        cont_vars_with_none = [""] + self.continuous_vars

        self.x_combo['values'] = all_vars_for_x
        self.y_combo['values'] = cont_vars_with_none
        self.hue_combo['values'] = cat_vars_with_none
        
        # Limpiar selecciones si la variable ya no existe o es inválida
        if not self.x_var.get() in all_vars_for_x: self.x_var.set("")
        if not self.y_var.get() in cont_vars_with_none: self.y_var.set("")
        if not self.hue_var.get() in cat_vars_with_none: self.hue_var.set("")
        
        self.plot_generated = False 
        self.update_ui_element_states()

    def update_ui_element_states(self):
        # ... (sin cambios)
        has_data = self.df is not None and not self.df.empty
        active_combo_state = 'readonly' if has_data else tk.DISABLED
        entry_state = tk.NORMAL if has_data else tk.DISABLED

        self.plot_type_combo.config(state=active_combo_state)
        self.x_combo.config(state=active_combo_state)
        self.y_combo.config(state=active_combo_state)
        self.hue_combo.config(state=active_combo_state)
        
        self.plot_title_entry.config(state=entry_state)
        self.plot_xlabel_entry.config(state=entry_state)
        self.plot_ylabel_entry.config(state=entry_state)

        can_generate = has_data and bool(self.x_var.get())
        self.generate_plot_button.config(state=tk.NORMAL if can_generate else tk.DISABLED)
        self.save_plot_button.config(state=tk.NORMAL if self.plot_generated else tk.DISABLED)
        
        # Actualizar estado de la toolbar de matplotlib
        if hasattr(self, 'toolbar'): # Asegurarse que la toolbar exista
            # La toolbar se maneja internamente, pero podríamos querer deshabilitar acciones si no hay gráfico
            # Por ahora, su propio estado interno debería ser suficiente.
            pass


    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*"))
        )
        if not filepath: return
        try:
            self.original_df = pd.read_csv(filepath) # Guardar original
            self.df = self.original_df.copy()       # Trabajar con una copia
            messagebox.showinfo("Carga Exitosa", f"Dataset cargado: {self.df.shape[0]} filas, {self.df.shape[1]} columnas.")
            self.ax.clear()
            self.canvas.draw()
            self.update_info()
            self.update_variable_lists_and_combos()
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo: {e}")
            self.df = None
            self.original_df = None
            self.ax.clear()
            self.canvas.draw()
            self.update_info()
            self.update_variable_lists_and_combos()

    def generate_sample_data(self):
        self.original_df = generar_datos_ejemplo() # Guardar original
        self.df = self.original_df.copy()          # Trabajar con una copia
        messagebox.showinfo("Datos de Ejemplo", "Se han generado datos de ejemplo.")
        self.ax.clear() 
        self.canvas.draw()
        self.update_info()
        self.update_variable_lists_and_combos()

    def clean_data(self):
        if self.df is None: # O self.original_df si queremos limpiar el original
            messagebox.showwarning("Sin Datos", "No hay dataset cargado para limpiar.")
            return
        
        # Asegurarse de trabajar sobre la copia actual si ya hay filtros, o sobre el original si no
        target_df_for_cleaning = self.df if self.df is not None else self.original_df

        if target_df_for_cleaning is None: # Doble check
            messagebox.showwarning("Sin Datos", "No hay dataset cargado para limpiar.")
            return

        rows_before = len(target_df_for_cleaning)
        cleaned_df = target_df_for_cleaning.dropna() # Realiza la limpieza en una nueva variable
        
        # Actualizar self.df con el df limpiado
        # Si se quiere que la limpieza sea "destructiva" para filtros futuros,
        # también se podría actualizar self.original_df
        self.df = cleaned_df 
        # self.original_df = cleaned_df.copy() # Opcional: si la limpieza debe afectar al "estado original"

        rows_after = len(self.df)
        
        messagebox.showinfo("Limpieza de Datos", 
                            f"Se eliminaron {rows_before - rows_after} filas con valores NaN.\n"
                            f"Dataset actual: {rows_after} filas.")
        self.ax.clear()
        self.canvas.draw()
        self.update_info() # Actualizar info con self.df
        self.update_variable_lists_and_combos() # Re-evaluar variables con self.df

    def generate_selected_plot(self):
        if self.df is None or self.df.empty:
            messagebox.showwarning("Sin Datos", "No hay datos para graficar.")
            return

        plot_type = self.plot_type_var.get()
        x_col = self.x_var.get()
        y_col = self.y_var.get() if self.y_var.get() else None
        hue_col = self.hue_var.get() if self.hue_var.get() else None

        custom_title = self.plot_title_var.get()
        custom_xlabel = self.plot_xlabel_var.get()
        custom_ylabel = self.plot_ylabel_var.get()

        if not x_col:
            messagebox.showwarning("Selección Requerida", "Por favor, seleccione al menos la Variable X.")
            return

        self.ax.clear()
        self.plot_generated = False 

        try:
            df_to_plot = self.df # Usar el dataframe actual (puede estar filtrado en el futuro)

            # Lógica para manejar muchas categorías en el eje X (ej. para gráficos de barras)
            # Esto es una heurística simple. Se puede mejorar.
            max_categories_display = 30 # Número máximo de categorías a mostrar directamente
            is_x_categorical = x_col in self.categorical_vars
            
            if is_x_categorical and plot_type == "Barra" and df_to_plot[x_col].nunique() > max_categories_display:
                # Si hay demasiadas categorías para un barplot, tomar las N más frecuentes
                # o advertir al usuario. Aquí, tomaremos las N más frecuentes.
                top_n = df_to_plot[x_col].value_counts().nlargest(max_categories_display).index
                df_to_plot = df_to_plot[df_to_plot[x_col].isin(top_n)]
                if not custom_title: # Añadir una nota al título si no hay uno personalizado
                    custom_title = f"(Mostrando Top {max_categories_display} para {x_col})"
                else:
                    custom_title += f" (Top {max_categories_display} de {x_col})"


            valid_plot = True
            if plot_type in ["Barra", "Caja (Box)", "Violín"]:
                if not y_col:
                    messagebox.showwarning("Selección Requerida", f"Gráfico {plot_type} requiere Variable Y.")
                    valid_plot = False
            elif plot_type == "Histograma":
                 pass 

            if not valid_plot:
                self.update_ui_element_states()
                return

            # ... (llamadas a viz_graficos sin cambios, pero usando df_to_plot) ...
            if plot_type == "Barra":
                viz_graficos.crear_grafico_barras(df_to_plot, self.ax, x_col, y_col, hue_col,
                                                  title=custom_title, xlabel=custom_xlabel, ylabel=custom_ylabel)
            elif plot_type == "Caja (Box)":
                viz_graficos.crear_boxplot(df_to_plot, self.ax, x_col, y_col, hue_col,
                                           title=custom_title, xlabel=custom_xlabel, ylabel=custom_ylabel)
            elif plot_type == "Violín":
                viz_graficos.crear_violinplot(df_to_plot, self.ax, x_col, y_col, hue_col,
                                             title=custom_title, xlabel=custom_xlabel, ylabel=custom_ylabel)
            elif plot_type == "Histograma":
                viz_graficos.crear_histograma(df_to_plot, self.ax, x_col, hue_col,
                                             title=custom_title, xlabel=custom_xlabel, ylabel=custom_ylabel)
            else:
                messagebox.showerror("Error", "Tipo de gráfico no reconocido.")
                return

            
            self.ax.tick_params(axis='x', labelrotation=45)
            # Ajuste para etiquetas del eje X si son muchas
            # Esto es más complejo de generalizar bien sin conocer la naturaleza de los datos
            # Por ahora, la rotación es la principal ayuda. El zoom/paneo ayudará más.
            current_xticks = self.ax.get_xticks()
            if len(current_xticks) > 40 and plot_type == "Barra": # Si hay muchísimos ticks en un barplot
                # Podríamos intentar mostrar uno de cada N ticks, pero esto requiere cuidado
                # Ejemplo: self.ax.set_xticks(current_xticks[::5]) # Muestra 1 de cada 5
                # Por ahora, nos fiaremos de la rotación y el zoom/paneo.
                pass

            for label in self.ax.get_xticklabels():
                if label: label.set_horizontalalignment('right')
            
            # Intentar un ajuste más agresivo si hay problemas de superposición
            try:
                self.fig.tight_layout(pad=1.5) # Aumentar padding
            except ValueError: # tight_layout puede fallar a veces
                 self.fig.subplots_adjust(bottom=0.15, left=0.15, right=0.95, top=0.9) # Ajuste manual


            self.canvas.draw()
            self.plot_generated = True

        except Exception as e:
            messagebox.showerror("Error al Graficar", f"Ocurrió un error: {e}\n"
                                 "Verifique la selección de variables, sus tipos, y los datos.")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error al graficar:\n{e}", ha='center', va='center', wrap=True, color='red')
            self.canvas.draw()
            self.plot_generated = False
        
        self.update_ui_element_states()


    def save_plot(self):
        # ... (sin cambios)
        if not self.plot_generated:
            messagebox.showwarning("Sin Gráfico", "No hay gráfico generado para guardar.")
            return

        filepath = filedialog.asksaveasfilename(
            title="Guardar Gráfico Como",
            filetypes=(("PNG files", "*.png"),
                       ("JPEG files", "*.jpg;*.jpeg"),
                       ("SVG files", "*.svg"),
                       ("PDF files", "*.pdf"),
                       ("All files", "*.*")),
            defaultextension=".png"
        )
        if not filepath:
            return
        
        try:
            # Guardar con fondo blanco por defecto para mejor portabilidad
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='w')
            messagebox.showinfo("Gráfico Guardado", f"Gráfico guardado en: {filepath}")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el gráfico: {e}")

    def update_info(self):
        # ... (sin cambios)
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        
        display_df = self.df if self.df is not None else pd.DataFrame() # Usar un df vacío si no hay datos

        if not display_df.empty:
            info_str = f"Dimensiones (dataset actual): {display_df.shape[0]} filas, {display_df.shape[1]} columnas\n\n"
            self.info_text.insert(tk.END, info_str)
            
            buffer = io.StringIO()
            display_df.info(buf=buffer)
            self.info_text.insert(tk.END, "--- Información General (df.info()) ---\n" + buffer.getvalue() + "\n\n")

            self.info_text.insert(tk.END, "--- Resumen de Nulos (df.isnull().sum()) ---\n" + display_df.isnull().sum().to_string() + "\n\n")
            
            try: 
                desc_summary = display_df.describe(include='all').to_string()
                self.info_text.insert(tk.END, "--- Estadísticas Descriptivas (df.describe(include='all')) ---\n" + desc_summary)
            except Exception as e:
                self.info_text.insert(tk.END, f"--- Estadísticas Descriptivas ---\nError al generar descripción: {e}")
        else:
            self.info_text.insert(tk.END, "No hay dataset cargado o el dataset actual está vacío.")
        self.info_text.config(state=tk.DISABLED)