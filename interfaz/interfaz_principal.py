import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype # For ordered categorical plotting
import numpy as np
import io

from datos.generador_datos import generar_datos_ejemplo
import visualizacion.graficos as viz_graficos

class DiabetesRiskAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Analizador de Datos Exploratorio")
        self.root.geometry("1500x900")
        self.root.configure(bg='#f0f0f0')
        
        self.df = None
        self.original_df = None
        self.categorical_vars = []
        self.continuous_vars = []
        self.plot_generated = False

        # Paging configuration
        self.X_AXIS_CATEGORY_THRESHOLD = 25 
        self.x_page_chunk_size = 20        
        self.current_x_page = 0
        self.total_x_pages = 0
        self.x_page_values = []            
        self.last_paged_x_var = None       
        self.last_plot_type_for_paging = None


        self.setup_style()
        self.create_interface()
        self.update_variable_lists_and_combos() 

    def setup_style(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0', foreground='#2c3e50')
        style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#f0f0f0', foreground='#34495e')
        style.configure('Modern.TButton', font=('Arial', 10), padding=(10, 5))
        style.configure('Small.Modern.TButton', font=('Arial', 9), padding=(5, 3))


    def create_interface(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=3)
        main_frame.columnconfigure(2, weight=5)
        main_frame.rowconfigure(1, weight=1)

        title_label = ttk.Label(main_frame, text="📊 Analizador de Datos Exploratorio", style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky=tk.W)

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
        self.plot_type_combo.bind("<<ComboboxSelected>>", self.on_plot_type_changed)

        ttk.Label(left_frame, text="Variable X:").grid(row=13, column=0, sticky=tk.W, pady=1)
        self.x_var = tk.StringVar()
        self.x_var.trace_add("write", self.on_x_variable_selection_change)
        self.x_combo = ttk.Combobox(left_frame, textvariable=self.x_var, state="readonly")
        self.x_combo.grid(row=14, column=0, sticky=(tk.W, tk.E), pady=1)

        ttk.Label(left_frame, text="Variable Y (si aplica):").grid(row=15, column=0, sticky=tk.W, pady=1)
        self.y_var = tk.StringVar()
        self.y_var.trace_add("write", self.on_generic_variable_selection_change) 
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
        
        # Paging controls
        self.paging_frame = ttk.Frame(left_frame) # Store as self.paging_frame
        self.paging_frame.grid(row=25, column=0, sticky=(tk.W, tk.E), pady=(5,0))
        self.paging_frame.columnconfigure(0, weight=1) 
        self.paging_frame.columnconfigure(1, weight=0) 
        self.paging_frame.columnconfigure(2, weight=0) 

        self.paging_label = ttk.Label(self.paging_frame, text="")
        self.paging_label.grid(row=0, column=0, sticky=tk.W, padx=(0,5))
        
        self.prev_page_button = ttk.Button(self.paging_frame, text="< X", command=self.prev_x_page, style='Small.Modern.TButton')
        self.prev_page_button.grid(row=0, column=1, sticky=tk.E, padx=2)
        
        self.next_page_button = ttk.Button(self.paging_frame, text="X >", command=self.next_x_page, style='Small.Modern.TButton')
        self.next_page_button.grid(row=0, column=2, sticky=tk.E)


        self.generate_plot_button = ttk.Button(left_frame, text="📊 Generar Gráfico", 
            command=self.generate_selected_plot, style='Modern.TButton')
        self.generate_plot_button.grid(row=26, column=0, sticky=(tk.W, tk.E), pady=(5,2))

        self.save_plot_button = ttk.Button(left_frame, text="💾 Guardar Gráfico",
                                   command=self.save_plot, style='Modern.TButton')
        self.save_plot_button.grid(row=27, column=0, sticky=(tk.W, tk.E), pady=2)

        center_frame = ttk.LabelFrame(main_frame, text="Información del Dataset", padding="10")
        center_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        center_frame.rowconfigure(0, weight=1)
        center_frame.columnconfigure(0, weight=1)
        self.info_text = scrolledtext.ScrolledText(center_frame, width=50, height=25, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        right_frame = ttk.LabelFrame(main_frame, text="Vista Previa de Gráficos", padding="10")
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        right_frame.rowconfigure(0, weight=0) 
        right_frame.rowconfigure(1, weight=1) 
        right_frame.columnconfigure(0, weight=1)
        
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, right_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.grid(row=0, column=0, sticky=tk.EW)

        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def _reset_paging_state(self):
        self.current_x_page = 0
        self.total_x_pages = 0
        self.x_page_values = []
        self.last_paged_x_var = None
        self.last_plot_type_for_paging = None

    def on_plot_type_changed(self, event=None):
        self._reset_paging_state()
        self._configure_plot_variable_combos()

    def on_x_variable_selection_change(self, name, index, mode):
        self._reset_paging_state()
        self.update_ui_element_states()

    def on_generic_variable_selection_change(self, name, index, mode):
        self.update_ui_element_states()

    def update_variable_lists_and_combos(self):
        self._reset_paging_state()
        self.cat_listbox.delete(0, tk.END)
        self.cont_listbox.delete(0, tk.END)
        
        self.categorical_vars = []
        self.continuous_vars = []

        if self.original_df is not None:
            self.df = self.original_df.copy() 
        else: # Ensure self.df is cleared if original_df is None
            self.df = None

        if self.df is not None:
            for col in self.df.columns:
                if self.df[col].isnull().all():
                    continue 

                col_dtype = self.df[col].dtype
                if col_dtype in ['object', 'string', 'bool'] or pd.api.types.is_categorical_dtype(col_dtype):
                    self.categorical_vars.append(col)
                elif pd.api.types.is_numeric_dtype(col_dtype):
                    if self.df[col].nunique() < self.X_AXIS_CATEGORY_THRESHOLD and \
                       len(self.df) > 50 and self.df[col].nunique() > 1: 
                        try:
                            # Check if all unique non-NaN values are integer-like
                            is_int_like = all(float(x).is_integer() for x in self.df[col].dropna().unique())
                        except (ValueError, TypeError): 
                            is_int_like = False

                        if is_int_like:
                             self.categorical_vars.append(col)
                        else:
                            self.continuous_vars.append(col)
                    else: 
                        self.continuous_vars.append(col)
                elif pd.api.types.is_datetime64_any_dtype(col_dtype):
                     self.categorical_vars.append(col)
                else: 
                    self.categorical_vars.append(col)
            
            self.categorical_vars = sorted(list(set(self.categorical_vars)))
            self.continuous_vars = sorted(list(set(c for c in self.continuous_vars if c not in self.categorical_vars)))

            for var in self.categorical_vars:
                self.cat_listbox.insert(tk.END, var)
            for var in self.continuous_vars:
                self.cont_listbox.insert(tk.END, var)
        
        self._configure_plot_variable_combos() 
        self.plot_generated = False 

    def _configure_plot_variable_combos(self):
        current_plot_type = self.plot_type_var.get()
        
        cat_vars_with_none = [""] + self.categorical_vars
        cont_vars_with_none = [""] + self.continuous_vars
        all_vars_with_none = [""] + sorted(list(set(self.categorical_vars + self.continuous_vars)))

        x_options, y_options = all_vars_with_none, all_vars_with_none 
        y_combo_state = "readonly"

        if current_plot_type == "Barra":
            x_options = all_vars_with_none # Allow numeric X for bar (seaborn treats as categorical)
            y_options = cont_vars_with_none
        elif current_plot_type in ["Caja (Box)", "Violín"]:
            x_options = all_vars_with_none # Allow numeric X (seaborn treats as categorical)
            y_options = cont_vars_with_none
        elif current_plot_type == "Histograma":
            x_options = cont_vars_with_none
            y_options = [""] 
            self.y_var.set("") 
            y_combo_state = tk.DISABLED
        
        self.x_combo['values'] = x_options
        self.y_combo['values'] = y_options
        self.hue_combo['values'] = cat_vars_with_none

        if self.x_var.get() not in self.x_combo['values']: self.x_var.set("")
        if self.y_var.get() not in self.y_combo['values']: self.y_var.set("")
        if self.hue_var.get() not in self.hue_combo['values']: self.hue_var.set("")
        
        self.y_combo.config(state=y_combo_state if self.df is not None and not self.df.empty else tk.DISABLED)
        self.update_ui_element_states()


    def update_ui_element_states(self):
        has_data = self.df is not None and not self.df.empty
        current_plot_type = self.plot_type_var.get()

        base_combo_state = 'readonly' if has_data else tk.DISABLED
        entry_state = tk.NORMAL if has_data else tk.DISABLED

        self.plot_type_combo.config(state=base_combo_state)
        self.x_combo.config(state=base_combo_state)
        
        if not has_data:
            self.y_combo.config(state=tk.DISABLED)
            self.hue_combo.config(state=tk.DISABLED)
        else:
            if self.y_combo.cget('state') != tk.DISABLED : 
                 self.y_combo.config(state='readonly')
            self.hue_combo.config(state='readonly')


        self.plot_title_entry.config(state=entry_state)
        self.plot_xlabel_entry.config(state=entry_state)
        self.plot_ylabel_entry.config(state=entry_state)

        can_generate = False
        if has_data and bool(self.x_var.get()):
            x_is_cont = self.x_var.get() in self.continuous_vars
            # x_is_cat = self.x_var.get() in self.categorical_vars # Not strictly needed here
            
            if current_plot_type == "Histograma":
                if x_is_cont: # Histogram needs continuous X
                    can_generate = True
            elif current_plot_type in ["Barra", "Caja (Box)", "Violín"]:
                if bool(self.y_var.get()) and self.y_var.get() in self.continuous_vars: # Y must be selected and continuous
                     # X can be categorical or continuous (treated as discrete by seaborn)
                    can_generate = True
        
        self.generate_plot_button.config(state=tk.NORMAL if can_generate else tk.DISABLED)
        self.save_plot_button.config(state=tk.NORMAL if self.plot_generated else tk.DISABLED)
        
        # Paging controls update
        can_page_prev = self.total_x_pages > 0 and self.current_x_page > 0
        can_page_next = self.total_x_pages > 0 and self.current_x_page < self.total_x_pages - 1
        
        self.prev_page_button.config(state=tk.NORMAL if can_page_prev else tk.DISABLED)
        self.next_page_button.config(state=tk.NORMAL if can_page_next else tk.DISABLED)

        if self.total_x_pages > 0:
            self.paging_label.config(text=f"Page {self.current_x_page + 1}/{self.total_x_pages}")
            self.paging_frame.grid() 
        else:
            self.paging_label.config(text="")
            self.paging_frame.grid_remove()
            
        if hasattr(self, 'toolbar'):
            pass

    def load_data(self):
        filepath = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=(("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*"))
        )
        if not filepath: return
        try:
            self.original_df = pd.read_csv(filepath)
            # self.df is set in update_variable_lists_and_combos
            messagebox.showinfo("Carga Exitosa", f"Dataset cargado: {self.original_df.shape[0]} filas, {self.original_df.shape[1]} columnas.")
            self._reset_paging_state()
            self.ax.clear()
            self.canvas.draw()
            self.update_info() # Call before var lists to use potentially newly loaded self.df
            self.update_variable_lists_and_combos()
        except Exception as e:
            messagebox.showerror("Error de Carga", f"No se pudo cargar el archivo: {e}")
            self.original_df = None
            self.df = None
            self._reset_paging_state()
            self.ax.clear()
            self.canvas.draw()
            self.update_info()
            self.update_variable_lists_and_combos()

    def generate_sample_data(self):
        self.original_df = generar_datos_ejemplo(n_samples=10000) 
        # self.df is set in update_variable_lists_and_combos
        messagebox.showinfo("Datos de Ejemplo", "Se han generado datos de ejemplo (10000 filas).")
        self._reset_paging_state()
        self.ax.clear() 
        self.canvas.draw()
        self.update_info() 
        self.update_variable_lists_and_combos()

    def clean_data(self):
        if self.original_df is None:
            messagebox.showwarning("Sin Datos", "No hay dataset cargado para limpiar.")
            return
        
        rows_before = len(self.original_df)
        cleaned_df = self.original_df.dropna()
        num_removed = rows_before - len(cleaned_df)

        if num_removed == 0:
            messagebox.showinfo("Limpieza de Datos", "No se encontraron filas con valores NaN para eliminar.")
        else:
            messagebox.showinfo("Limpieza de Datos", 
                                f"Se eliminaron {num_removed} filas con valores NaN.\n"
                                f"Dataset original actualizado: {len(cleaned_df)} filas.")
        
        self.original_df = cleaned_df
        # self.df will be updated by update_variable_lists_and_combos

        self._reset_paging_state()
        self.ax.clear()
        self.canvas.draw()
        self.update_info() 
        self.update_variable_lists_and_combos() 

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
            messagebox.showwarning("Selección Requerida", "Por favor, seleccione la Variable X.")
            return
        if plot_type in ["Barra", "Caja (Box)", "Violín"] and not y_col:
            messagebox.showwarning("Selección Requerida", f"Gráfico {plot_type} requiere Variable Y.")
            return

        # Reset paging if X var or plot type (relevant to paging) has changed since last paging setup
        if self.last_paged_x_var != x_col or self.last_plot_type_for_paging != plot_type:
            self._reset_paging_state()
            self.last_paged_x_var = x_col
            self.last_plot_type_for_paging = plot_type
        
        self.ax.clear()
        self.plot_generated = False 

        paging_active_for_this_plot = False
        # (Re)-calculate paging parameters if not already set up for current x_col and plot_type
        if self.total_x_pages == 0 and x_col: 
            if plot_type in ["Barra", "Caja (Box)", "Violín"]:
                unique_x_vals_for_paging = sorted(self.df[x_col].dropna().unique())
                num_unique_x = len(unique_x_vals_for_paging)

                if num_unique_x > self.X_AXIS_CATEGORY_THRESHOLD:
                    paging_active_for_this_plot = True
                    self.x_page_values = unique_x_vals_for_paging
                    self.total_x_pages = (num_unique_x + self.x_page_chunk_size - 1) // self.x_page_chunk_size
                    # self.current_x_page is already 0 if reset occurred
            else: # Plot type not eligible for paging
                 self._reset_paging_state() # Clear any previous paging setup
        elif self.total_x_pages > 0 and x_col == self.last_paged_x_var and plot_type == self.last_plot_type_for_paging:
             paging_active_for_this_plot = True # Paging already set up and context is the same


        try:
            df_to_plot = self.df.copy() 

            if paging_active_for_this_plot:
                start_idx = self.current_x_page * self.x_page_chunk_size
                end_idx = min(start_idx + self.x_page_chunk_size, len(self.x_page_values))
                current_page_x_subset = self.x_page_values[start_idx:end_idx]
                
                df_to_plot = df_to_plot[df_to_plot[x_col].isin(current_page_x_subset)]
                
                # Ensure x_col in df_to_plot is ordered according to current_page_x_subset for seaborn
                cat_type = CategoricalDtype(categories=current_page_x_subset, ordered=True)
                df_to_plot[x_col] = df_to_plot[x_col].astype(cat_type)

                title_suffix = f" ({x_col} - Page {self.current_x_page + 1}/{self.total_x_pages})"
                custom_title = f"{custom_title}{title_suffix}" if custom_title else title_suffix.strip()

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
                self.update_ui_element_states()
                return

            self.ax.tick_params(axis='x', rotation=45)
            # Ensure horizontal alignment is right for rotated labels
            for label in self.ax.get_xticklabels():
                if label: label.set_horizontalalignment('right')
            
            try:
                self.fig.tight_layout(pad=1.5)
            except (ValueError, RuntimeError): # RuntimeError can also occur with tight_layout
                 self.fig.subplots_adjust(bottom=0.25, left=0.15, right=0.95, top=0.9) 

            self.canvas.draw()
            self.plot_generated = True

        except Exception as e:
            messagebox.showerror("Error al Graficar", f"Ocurrió un error: {e}\n"
                                 "Verifique la selección de variables, sus tipos, y los datos.")
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Error al graficar:\n{str(e)[:200]}...", # Limit error message length
                         ha='center', va='center', wrap=True, color='red', fontsize=9)
            self.canvas.draw()
            self.plot_generated = False
        
        self.update_ui_element_states()

    def prev_x_page(self):
        if self.total_x_pages > 0 and self.current_x_page > 0:
            self.current_x_page -= 1
            self.generate_selected_plot()

    def next_x_page(self):
        if self.total_x_pages > 0 and self.current_x_page < self.total_x_pages - 1:
            self.current_x_page += 1
            self.generate_selected_plot()

    def save_plot(self):
        if not self.plot_generated:
            messagebox.showwarning("Sin Gráfico", "No hay gráfico generado para guardar.")
            return
        filepath = filedialog.asksaveasfilename(
            title="Guardar Gráfico Como",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"),
                       ("SVG files", "*.svg"), ("PDF files", "*.pdf"), ("All files", "*.*")),
            defaultextension=".png"
        )
        if not filepath: return
        try:
            self.fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='w')
            messagebox.showinfo("Gráfico Guardado", f"Gráfico guardado en: {filepath}")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el gráfico: {e}")

    def update_info(self):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        
        # Use self.df for info display, which is a copy of original_df (possibly cleaned)
        # or None if no data is loaded.
        display_df = self.df if self.df is not None else pd.DataFrame()

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

if __name__ == '__main__': # Should be in main.py, but for testing here
    root = tk.Tk()
    app = DiabetesRiskAnalyzer(root)
    root.mainloop()