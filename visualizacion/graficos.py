import seaborn as sns
import matplotlib.pyplot as plt

def crear_grafico_barras(df, ax, col_categorica, col_numerica, col_hue=None, title=None, xlabel=None, ylabel=None):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No hay datos para graficar.", ha='center', va='center')
        return
    sns.barplot(data=df, x=col_categorica, y=col_numerica, hue=col_hue, ax=ax, errorbar='sd', capsize=.1)
    ax.set_title(title if title else f"Media de {col_numerica} por {col_categorica}" + (f" (agrupado por {col_hue})" if col_hue else ""))
    ax.set_xlabel(xlabel if xlabel else col_categorica)
    ax.set_ylabel(ylabel if ylabel else f"Media de {col_numerica} (±DE)")

def crear_boxplot(df, ax, col_categorica, col_numerica, col_hue=None, title=None, xlabel=None, ylabel=None):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No hay datos para graficar.", ha='center', va='center')
        return
    sns.boxplot(data=df, x=col_categorica, y=col_numerica, hue=col_hue, ax=ax)
    ax.set_title(title if title else f"Distribución de {col_numerica} por {col_categorica}" + (f" (agrupado por {col_hue})" if col_hue else ""))
    ax.set_xlabel(xlabel if xlabel else col_categorica)
    ax.set_ylabel(ylabel if ylabel else col_numerica)

def crear_violinplot(df, ax, col_categorica, col_numerica, col_hue=None, title=None, xlabel=None, ylabel=None):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No hay datos para graficar.", ha='center', va='center')
        return
    sns.violinplot(data=df, x=col_categorica, y=col_numerica, hue=col_hue, ax=ax)
    ax.set_title(title if title else f"Densidad de {col_numerica} por {col_categorica}" + (f" (agrupado por {col_hue})" if col_hue else ""))
    ax.set_xlabel(xlabel if xlabel else col_categorica)
    ax.set_ylabel(ylabel if ylabel else col_numerica)

def crear_histograma(df, ax, col_numerica, col_hue_cat=None, title=None, xlabel=None, ylabel=None):
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No hay datos para graficar.", ha='center', va='center')
        return
    sns.histplot(data=df, x=col_numerica, hue=col_hue_cat, multiple='stack', ax=ax, kde=True)
    ax.set_title(title if title else f"Distribución de {col_numerica}" + (f" (agrupado por {col_hue_cat})" if col_hue_cat else ""))
    ax.set_xlabel(xlabel if xlabel else col_numerica)
    ax.set_ylabel(ylabel if ylabel else "Frecuencia")