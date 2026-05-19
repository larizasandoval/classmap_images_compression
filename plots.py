"""
Script para generar los gráficos de análisis
Created by Lariza Sandoval, Mayo 2026.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm


# 1. Configuración de estilos académicos (Fuentes tipográficas y tamaños)
plt.rcParams.update({
        'font.family': 'serif',        # Usa fuentes tipo Times New Roman
        'font.size': 10,               # Tamaño estándar para figuras de 1 columna
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'text.usetex': False           # Cambiarlo a True si cuando tenga LaTeX instalado 
    })

def plot_entropy_comparison(entropy_results_df, details=True):

 
    if details:
        df_renamed = entropy_results_df.rename(columns={
            "H_limit_bits": "Limit entropy — decision bits",
            "bpp_bits": "Bit rate — decision bits",
            "H_limit_raw": "Theoretical entropy — raw values",
            "bpp_raw": "Real rate — raw (uncoded) values"
        })
        # Transformación de datos a formato largo
        df_melted = df_renamed.melt(
            id_vars=["n_clases"],
            value_vars=["Limit entropy — decision bits", "Bit rate — decision bits", "Theoretical entropy — raw values", "Real rate — raw (uncoded) values"],
            var_name="Entropy Metric",
            value_name="Value"
        )
    else:
        df_renamed = entropy_results_df.rename(columns={
            "H_bits": "Decision bits without context model",
            "H_limit_bits": "Decision bits with context model",
        })
        # Transformación de datos a formato largo
        df_melted = df_renamed.melt(
            id_vars=["n_clases"],
            value_vars=["Decision bits without context model", "Decision bits with context model"],
            var_name="Entropy Metric",
            value_name="Value"
        )
    # 3. Creación del lienzo con dimensiones típicas de columna científica (en pulgadas)
    fig, ax = plt.subplots(figsize=(6.5, 4)) 
    sns.set_style("white") # Fondo blanco puro sin rejillas por defecto

    # Paleta de colores sobria y apta para daltónicos/impresión
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # Dibujar las barras con bordes definidos
    ax = sns.barplot(
        data=df_melted,
        x=df_melted["n_clases"].astype(str),
        y="Value",
        hue="Entropy Metric",
        palette=colors,
        edgecolor="black", # Bordes negros delgados en cada barra
        linewidth=0.8
    )

    # 4. Detalles académicos del Gráfico
    # Añadimos líneas horizontales discontinuas y muy sutiles para ayudar a leer valores
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True) # Hace que las líneas de la cuadrícula queden por detrás de las barras

    # Etiquetas usando notación matemática para la variable del eje X (subíndice)
    ax.set_xlabel("Number of Classes ($n_{clases}$)")
    ax.set_ylabel("bpp")

    # Leyenda con recuadro limpio y dentro del gráfico
    ax.legend(title="Metrics", loc="upper left", frameon=True, edgecolor="black", framealpha=0.9)

    # Remover los bordes superior y derecho redundantes (Estilo "Despine")
    sns.despine()

    plt.tight_layout()

    # 5. Guardar en PDF y PNG
    if details:
        plt.savefig("analisis/analysis_entropy_plot.pdf", format="pdf", dpi=300)
        plt.savefig("analisis/analysis_entropy_plot.png", format="png", dpi=300)
    else: 
        plt.savefig("analisis/entropy_context_comparison_plot.pdf", format="pdf", dpi=300)
        plt.savefig("analisis/entropy_context_comparison_plot.png", format="png", dpi=300)
    
    plt.show()

def plot_residual_stats(residual_stats_df):

    df = residual_stats_df.sort_values(by="n_clases").reset_index(drop=True)

    # 2. Definición de variables y etiquetas limpias en inglés
    categories = ["Resolved in 1 decision", "Resolved in 2 decisions", "Resolved in 3 decisions", "Resolved in 4 decisions", "Unmatched(raw value)"]
    columns_to_plot = ["pct_1bit", "pct_2bits", "pct_3bits", "pct_4bits", "pct_unmatched"]
    x_labels = df["n_clases"].astype(str)

    # 3. Creación del lienzo
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sns.set_style("white")

    # Paleta secuencial/cualitativa sobria y distinguible en escala de grises
    colors = ["#4c72b0", "#55a868", "#dd8452", "#c44e52", "#8172b3"]

    # 4. Construcción de las barras apiladas (Stacked Bar)
    # En Matplotlib, el truco consiste en usar el argumento 'bottom' para apilar los datos previos
    bottoms = [0] * len(df)

    for col, cat_name, color in zip(columns_to_plot, categories, colors):
        ax.bar(
            x_labels, 
            df[col], 
            bottom=bottoms, 
            label=cat_name, 
            color=color, 
            edgecolor="black", 
            linewidth=0.7,
            width=0.55  # Ajuste del ancho de barra para estética limpia
        )
        # Acumular la altura de la barra actual para servir de base a la siguiente
        bottoms = [b + v for b, v in zip(bottoms, df[col])]

    # 5. Detalles de formato académico
    # Cuadrícula sutil solo en el eje Y
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True)

    # Forzar el límite en 100% si los datos se aproximan numéricamente por el redondeo
    ax.set_ylim(0, 100)

    # Formatear el eje Y agregando el símbolo de porcentaje de manera limpia
    ax.yaxis.set_major_formatter(lambda x, pos: f'{int(x)}%')

    # Etiquetas del gráfico
    ax.set_xlabel("Number of Classes ($n_{clases}$)")
    ax.set_ylabel("Percentage of Pixels")

    # Leyenda invertida para que el orden visual coincida con el apilamiento de abajo hacia arriba
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reversed(handles), 
        reversed(labels), 
        title="Decision sequence length", 
        loc="upper left", 
        bbox_to_anchor=(1.02, 1), # Se saca ligeramente a la derecha para no obstruir las barras
        frameon=True, 
        edgecolor="black", 
        framealpha=1
    )

    # Quitar bordes innecesarios
    sns.despine()

    plt.tight_layout()

    # 6. Exportación 
    plt.savefig("analisis/stacked_decision_bits_plot.pdf", format="pdf", dpi=300)
    plt.savefig("analisis/stacked_decision_bits_plot.png", format="png", dpi=300)
    plt.show()


def plot_context_model_analysis(context_model_analysis_df):
                        
    # 1. Pivotar y limpiar índices
    df_pivot = context_model_analysis_df.pivot(
        index="n_clases", columns="contexto", values="total_decisions"
    )

    # Ordenar numéricamente los ejes para que el paper sea riguroso
    df_pivot = df_pivot.sort_index(ascending=True)  # Eje Y: Menor a mayor clase
    df_pivot = df_pivot.reindex(
        columns=sorted(df_pivot.columns)
    )  # Eje X: Menor a mayor contexto

    # 2. Lienzo
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    sns.set_style("white")

    # 3. Crear el Heatmap con escala logarítmica y sin anotaciones internas
    ax = sns.heatmap(
        df_pivot,
        cmap="Blues",
        annot=False,  
        norm=LogNorm(),  # Escala logarítmica para balancear las diferencias de tamaño
        linewidths=0.5,
        linecolor="black",
        cbar_kws={
            "label": "Total decisions bits (log scale)",
            "pad": 0.02,
        },  # Especificamos que es logarítmica
    )

    # 4. Refinar detalles de los ejes
    ax.set_xlabel("Contexts", labelpad=8)
    ax.set_ylabel("$n_{clases}$", labelpad=8)

    # Ajustar ticks del eje X para que respiren (por ejemplo, mostrar de 2 en 2 si hay muchos)
    # Si son del 0 al 36 secuenciales, esto los dejará limpios:
    ax.set_xticks([i for i in range(len(df_pivot.columns))])
    ax.set_xticklabels(df_pivot.columns, rotation=45, ha="right")

    # Asegurar que el eje Y se lea perfectamente recto
    plt.yticks(rotation=0)

    plt.tight_layout()

    # 5. Guardar copias
    plt.savefig("analisis/contexts_distribution_analysis.pdf", format="pdf", dpi=300)
    plt.savefig("analisis/contexts_distribution_analysis.png", format="png", dpi=300)

    plt.show()