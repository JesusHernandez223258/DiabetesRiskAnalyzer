import seaborn as sns

def crear_graficos(df, ax):
    if df is None:
        return
    ax.clear()
    sns.histplot(data=df, x='BMI', hue='Sex', multiple='stack', ax=ax)
    ax.set_title("Distribución del IMC por Sexo")
