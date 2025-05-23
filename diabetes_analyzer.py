import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

class DiabetesRiskAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis de Factores de Riesgo - Diabetes Tipo 2")
        self.root.geometry("1400x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables de datos
        self.df = None
        self.categorical_vars = []
        self.continuous_vars = []
        
        # Configurar estilo
        self.setup_style()
        
        # Crear interfaz
        self.create_interface()
        
        # Quita esta línea:
        # self.generate_sample_data()
    
    def setup_style(self):
        """Configurar estilo moderno para la aplicación"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Colores modernos
        style.configure('Title.TLabel', 
                       font=('Arial', 16, 'bold'), 
                       background='#f0f0f0',
                       foreground='#2c3e50')
        
        style.configure('Header.TLabel', 
                       font=('Arial', 12, 'bold'), 
                       background='#f0f0f0',
                       foreground='#34495e')
        
        style.configure('Modern.TButton',
                       font=('Arial', 10),
                       padding=(10, 5))
        
        style.map('Modern.TButton',
                 background=[('active', '#3498db'),
                           ('!active', '#2980b9')])
    
    def create_interface(self):
        """Crear la interfaz principal"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Título
        title_label = ttk.Label(main_frame, 
                               text="🏥 Análisis de Factores de Riesgo - Diabetes Tipo 2",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Frame izquierdo - Controles
        left_frame = ttk.LabelFrame(main_frame, text="Controles", padding="10")
        left_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Botones de carga de datos
        ttk.Label(left_frame, text="Gestión de Datos", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        ttk.Button(left_frame, text="📁 Cargar Dataset", 
                  command=self.load_data, style='Modern.TButton').grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="🎲 Generar Datos de Ejemplo", 
                  command=self.generate_sample_data, style='Modern.TButton').grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="🧹 Limpiar Datos", 
                  command=self.clean_data, style='Modern.TButton').grid(row=3, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Separador
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Selección de variables
        ttk.Label(left_frame, text="Variables Categóricas", style='Header.TLabel').grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        
        self.cat_listbox = tk.Listbox(left_frame, height=4, selectmode=tk.MULTIPLE)
        self.cat_listbox.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(left_frame, text="Variables Continuas", style='Header.TLabel').grid(row=7, column=0, sticky=tk.W, pady=(0, 5))
        
        self.cont_listbox = tk.Listbox(left_frame, height=4, selectmode=tk.MULTIPLE)
        self.cont_listbox.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Separador
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).grid(row=9, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Botones de análisis
        ttk.Label(left_frame, text="Análisis y Visualización", style='Header.TLabel').grid(row=10, column=0, sticky=tk.W, pady=(0, 10))
        
        ttk.Button(left_frame, text="📊 Generar Gráficos de Barras", 
                  command=self.create_bar_plots, style='Modern.TButton').grid(row=11, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="📈 Generar Boxplots", 
                  command=self.create_boxplots, style='Modern.TButton').grid(row=12, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="🎻 Generar Violinplots", 
                  command=self.create_violinplots, style='Modern.TButton').grid(row=13, column=0, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Button(left_frame, text="📋 Generar Informe Completo", 
                  command=self.generate_complete_report, style='Modern.TButton').grid(row=14, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Frame central - Información de datos
        center_frame = ttk.LabelFrame(main_frame, text="Información del Dataset", padding="10")
        center_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self.info_text = scrolledtext.ScrolledText(center_frame, width=40, height=25)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame derecho - Visualización
        right_frame = ttk.LabelFrame(main_frame, text="Vista Previa de Gráficos", padding="10")
        right_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Canvas para matplotlib
        self.fig, self.ax = plt.subplots(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar pesos de grid
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        center_frame.columnconfigure(0, weight=1)
        center_frame.rowconfigure(0, weight=1)
        
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
    
    def generate_sample_data(self):
        """Generar datos de ejemplo para diabetes tipo 2"""
        np.random.seed(42)
        n_samples = 1000
        
        # Variables categóricas
        sex = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        ethnicity = np.random.choice(['Caucasian', 'Hispanic', 'African American', 'Asian'], 
                                   n_samples, p=[0.6, 0.2, 0.15, 0.05])
        smoking_status = np.random.choice(['Never', 'Former', 'Current'], 
                                        n_samples, p=[0.5, 0.3, 0.2])
        physical_activity = np.random.choice(['Low', 'Moderate', 'High'], 
                                           n_samples, p=[0.4, 0.4, 0.2])
        
        # Variables continuas (con correlaciones realistas)
        age = np.random.normal(50, 15, n_samples)
        age = np.clip(age, 18, 80)
        
        # BMI con tendencia por sexo
        bmi_base = np.where(sex == 'Male', 27, 26)
        bmi = np.random.normal(bmi_base, 4, n_samples)
        bmi = np.clip(bmi, 18, 45)
        
        # Glucosa en ayunas (correlacionada con BMI y edad)
        glucose_base = 85 + (bmi - 25) * 1.5 + (age - 45) * 0.3
        glucose = np.random.normal(glucose_base, 15, n_samples)
        glucose = np.clip(glucose, 70, 300)
        
        # Colesterol HDL (inversamente relacionado con BMI)
        hdl_base = np.where(sex == 'Female', 55, 45) - (bmi - 25) * 0.8
        hdl = np.random.normal(hdl_base, 10, n_samples)
        hdl = np.clip(hdl, 20, 100)
        
        # Consumo de alcohol
        alcohol = np.random.exponential(2, n_samples)
        alcohol = np.clip(alcohol, 0, 20)
        
        # Presión arterial sistólica
        systolic_base = 120 + (age - 45) * 0.5 + (bmi - 25) * 0.8
        systolic_bp = np.random.normal(systolic_base, 15, n_samples)
        systolic_bp = np.clip(systolic_bp, 90, 200)
        
        # Crear DataFrame
        self.df = pd.DataFrame({
            'Age': age,
            'Sex': sex,
            'BMI': bmi,
            'Ethnicity': ethnicity,
            'Smoking_Status': smoking_status,
            'Physical_Activity_Level': physical_activity,
            'Fasting_Blood_Glucose': glucose,
            'Cholesterol_HDL': hdl,
            'Alcohol_Consumption': alcohol,
            'Systolic_BP': systolic_bp
        })
        
        self.update_variable_lists()
        self.update_info_display()
        messagebox.showinfo("Éxito", "Datos de ejemplo generados correctamente")
    
    def load_data(self):
        """Cargar datos desde archivo CSV"""
        file_path = filedialog.askopenfilename(
            title="Seleccionar archivo CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.update_variable_lists()
                self.update_info_display()
                messagebox.showinfo("Éxito", f"Datos cargados correctamente\nFilas: {len(self.df)}, Columnas: {len(self.df.columns)}")
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar el archivo:\n{str(e)}")
    
    def update_variable_lists(self):
        """Actualizar las listas de variables categóricas y continuas"""
        if self.df is None:
            return
        
        # Limpiar listas
        self.cat_listbox.delete(0, tk.END)
        self.cont_listbox.delete(0, tk.END)
        
        # Identificar variables categóricas y continuas
        for col in self.df.columns:
            if self.df[col].dtype == 'object' or self.df[col].nunique() < 10:
                self.cat_listbox.insert(tk.END, col)
            else:
                self.cont_listbox.insert(tk.END, col)
    
    def update_info_display(self):
        """Actualizar la información del dataset"""
        if self.df is None:
            return
        
        info_text = "=== INFORMACIÓN DEL DATASET ===\n\n"
        info_text += f"Dimensiones: {self.df.shape[0]} filas x {self.df.shape[1]} columnas\n\n"
        
        info_text += "=== ESTADÍSTICAS DESCRIPTIVAS ===\n"
        info_text += str(self.df.describe()) + "\n\n"
        
        info_text += "=== VALORES FALTANTES ===\n"
        missing = self.df.isnull().sum()
        info_text += str(missing[missing > 0]) + "\n\n"
        
        info_text += "=== TIPOS DE DATOS ===\n"
        info_text += str(self.df.dtypes) + "\n\n"
        
        # Variables categóricas
        cat_vars = [col for col in self.df.columns if self.df[col].dtype == 'object' or self.df[col].nunique() < 10]
        info_text += "=== VARIABLES CATEGÓRICAS ===\n"
        for var in cat_vars:
            info_text += f"{var}: {list(self.df[var].unique())}\n"
        
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
    
    def clean_data(self):
        """Limpiar datos: manejar valores faltantes y outliers"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "No hay datos cargados")
            return
        
        original_shape = self.df.shape
        
        # Remover filas con demasiados valores faltantes
        self.df = self.df.dropna(thresh=len(self.df.columns) * 0.7)
        
        # Imputar valores faltantes
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Moda para categóricas
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col].fillna(mode_val[0], inplace=True)
            else:
                # Mediana para numéricas
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Remover outliers extremos (IQR method)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        self.update_info_display()
        messagebox.showinfo("Limpieza Completada", 
                           f"Datos limpiados exitosamente\n"
                           f"Antes: {original_shape}\n"
                           f"Después: {self.df.shape}")
    
    def get_selected_variables(self):
        """Obtener variables seleccionadas"""
        cat_selected = [self.cat_listbox.get(i) for i in self.cat_listbox.curselection()]
        cont_selected = [self.cont_listbox.get(i) for i in self.cont_listbox.curselection()]
        
        if not cat_selected or not cont_selected:
            messagebox.showwarning("Advertencia", "Selecciona al menos una variable categórica y una continua")
            return None, None
        
        return cat_selected, cont_selected
    
    def create_bar_plots(self):
        """Crear gráficos de barras con barras de error"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "No hay datos cargados")
            return
        
        cat_vars, cont_vars = self.get_selected_variables()
        if cat_vars is None:
            return
        
        # Crear figura con subplots
        n_plots = min(4, len(cat_vars) * len(cont_vars))
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for cat_var in cat_vars[:2]:
            for cont_var in cont_vars[:2]:
                if plot_idx >= 4:
                    break
                
                # Calcular medias y errores estándar
                grouped = self.df.groupby(cat_var)[cont_var].agg(['mean', 'std', 'count'])
                
                ax = axes[plot_idx]
                x_pos = range(len(grouped))
                
                bars = ax.bar(x_pos, grouped['mean'], 
                             yerr=grouped['std'], 
                             capsize=5,
                             color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(grouped)],
                             alpha=0.8)
                
                ax.set_xlabel(cat_var, fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{cont_var} (Media ± SD)', fontsize=12, fontweight='bold')
                ax.set_title(f'{cont_var} por {cat_var}', fontsize=14, fontweight='bold')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(grouped.index, rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Añadir valores en las barras
                for i, (bar, mean_val) in enumerate(zip(bars, grouped['mean'])):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + grouped['std'].iloc[i]/2,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontweight='bold')
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('graficos_barras.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        messagebox.showinfo("Éxito", "Gráficos de barras generados y guardados como 'graficos_barras.png'")
    
    def create_boxplots(self):
        """Crear boxplots"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "No hay datos cargados")
            return
        
        cat_vars, cont_vars = self.get_selected_variables()
        if cat_vars is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for cat_var in cat_vars[:2]:
            for cont_var in cont_vars[:2]:
                if plot_idx >= 4:
                    break
                
                ax = axes[plot_idx]
                
                # Crear boxplot con seaborn para mejor estética
                sns.boxplot(data=self.df, x=cat_var, y=cont_var, ax=ax,
                           palette=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
                
                ax.set_xlabel(cat_var, fontsize=12, fontweight='bold')
                ax.set_ylabel(cont_var, fontsize=12, fontweight='bold')
                ax.set_title(f'Distribución de {cont_var} por {cat_var}', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        messagebox.showinfo("Éxito", "Boxplots generados y guardados como 'boxplots.png'")
    
    def create_violinplots(self):
        """Crear violinplots"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "No hay datos cargados")
            return
        
        cat_vars, cont_vars = self.get_selected_variables()
        if cat_vars is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for cat_var in cat_vars[:2]:
            for cont_var in cont_vars[:2]:
                if plot_idx >= 4:
                    break
                
                ax = axes[plot_idx]
                
                # Crear violinplot
                sns.violinplot(data=self.df, x=cat_var, y=cont_var, ax=ax,
                              palette=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
                
                ax.set_xlabel(cat_var, fontsize=12, fontweight='bold')
                ax.set_ylabel(cont_var, fontsize=12, fontweight='bold')
                ax.set_title(f'Densidad de {cont_var} por {cat_var}', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('violinplots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        messagebox.showinfo("Éxito", "Violinplots generados y guardados como 'violinplots.png'")
    
    def generate_complete_report(self):
        """Generar informe completo con todos los gráficos"""
        if self.df is None:
            messagebox.showwarning("Advertencia", "No hay datos cargados")
            return
        
        cat_vars, cont_vars = self.get_selected_variables()
        if cat_vars is None:
            return
        
        # Crear informe en texto
        report = "=== INFORME DE ANÁLISIS DE FACTORES DE RIESGO - DIABETES TIPO 2 ===\n\n"
        
        report += f"Dataset: {self.df.shape[0]} observaciones, {self.df.shape[1]} variables\n\n"
        
        report += "=== VARIABLES ANALIZADAS ===\n"
        report += f"Variables Categóricas: {', '.join(cat_vars)}\n"
        report += f"Variables Continuas: {', '.join(cont_vars)}\n\n"
        
        report += "=== HALLAZGOS PRINCIPALES ===\n"
        
        # Análisis por variable categórica
        for cat_var in cat_vars:
            report += f"\n--- Análisis por {cat_var} ---\n"
            for cont_var in cont_vars:
                grouped = self.df.groupby(cat_var)[cont_var].agg(['mean', 'std', 'count'])
                report += f"\n{cont_var}:\n"
                for category, row in grouped.iterrows():
                    report += f"  {category}: {row['mean']:.2f} ± {row['std']:.2f} (n={row['count']})\n"
        
        report += "\n=== RECOMENDACIONES CLÍNICAS ===\n"
        report += "1. Identificar subgrupos con glucosa en ayunas > 125 mg/dL\n"
        report += "2. Focalizar intervenciones en grupos con BMI > 30\n"
        report += "3. Promover actividad física en grupos sedentarios\n"
        report += "4. Implementar programas de cesación tabáquica\n"
        report += "5. Monitoreo regular de colesterol HDL < 40 mg/dL\n"
        
        # Guardar informe
        with open('informe_diabetes.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Generar todos los gráficos
        self.create_bar_plots()
        self.create_boxplots()
        self.create_violinplots()
        
        messagebox.showinfo("Informe Completo", 
                           "Informe completo generado exitosamente:\n"
                           "- informe_diabetes.txt\n"
                           "- graficos_barras.png\n"
                           "- boxplots.png\n"
                           "- violinplots.png")

def main():
    root = tk.Tk()
    app = DiabetesRiskAnalyzer(root)
    
    # Ejecutar generación de datos *después* de que mainloop esté activo
    root.after(100, app.generate_sample_data)

    root.mainloop()

if __name__ == "__main__":
    main()