"""
Script para generar los gráficos de análisis
Created by Lariza Sandoval, May 2026.
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
            #"H_limit_bits": "Limit entropy — decision bits",
            #"bpp_bits": "Bit rate — decision bits",
            "Exp_bit_rate_raw": "Expected bit rate — raw values",
            "bit_rate_raw": "Real bit rate — raw (uncoded) values"
        })
        # Transformación de datos a formato largo
        df_melted = df_renamed.melt(
            id_vars=["n_clases"],
            value_vars=["Expected bit rate — raw values", "Real bit rate — raw (uncoded) values"],
            var_name="Entropy Metric",
            value_name="Value"
        )
    else:

        df_renamed = entropy_results_df.rename(columns={
            "H_seq": "Sequence decision bits",
            #'H_seq_cxt': "Sequence decision bit + ctx",
            "Exp_bit_rate_only_bits_seq": "Decision bits without context model",
            "Exp_bit_rate_bits": "Decision bits with context model",
        })
        # Transformación de datos a formato largo
        df_melted = df_renamed.melt(
            id_vars=["n_clases"],
            value_vars=["Sequence decision bits", "Decision bits without context model", "Decision bits with context model"],
            var_name="Entropy Metric",
            value_name="Value"
        )
    
    0

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
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
    ax.set_axisbelow(True) # Hace que las líneas de la cuadrícula queden por detrás de las barras

    # Etiquetas usando notación matemática para la variable del eje X (subíndice)
    ax.set_xlabel("Number of Classes ($n_{clases}$)")
    ax.set_ylabel(f"Compressed data rate (bps)")

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

def plot_context_distribution_profiles(context_model_analysis_df):
    """
    Genera gráficos de perfil (líneas/barras) para la distribución de contextos,
    creando una versión lineal y otra semilogarítmica para cada n_clases.
    """
    # 1. Asegurar el orden numérico de las clases
    unique_classes = sorted(context_model_analysis_df["n_clases"].unique())
    
    for n_clase in unique_classes:
        # Filtrar datos para la clase actual y ordenar por contexto
        df_clase = context_model_analysis_df[context_model_analysis_df["n_clases"] == n_clase]
        df_clase = df_clase.sort_values(by="contexto")
        
        contexts = df_clase["contexto"].astype(str).tolist()
        values = df_clase["total_decisions"].tolist()
        
        # Saltamos si no hay datos para esta clase
        if not values:
            continue

        # 2. Generar ambos tipos de escala: 'linear' y 'log'
        for scale in ["linear", "log"]:
            fig, ax = plt.subplots(figsize=(6.5, 3.8))
            sns.set_style("white")
            
            # Dibujar la línea con marcadores para que sea altamente legible
            ax.plot(
                contexts, 
                values, 
                marker='o', 
                markersize=4, 
                color="#4c72b0", 
                linewidth=1.5, 
                label=f"$n_{{clases}} = {n_clase}$"
            )
            
            # Relleno sutil debajo de la curva (opcional, le da un toque limpio)
            ax.fill_between(contexts, values, color="#4c72b0", alpha=0.1)
            
            # 3. Configuración de la escala del eje Y
            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylabel("Total decision bits (log scale)")
                scale_suffix = "semilog"
            else:
                ax.set_ylabel("Total decision bits (linear)")
                scale_suffix = "linear"
                
            # 4. Detalles estéticos académicos
            ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='#cccccc')
            ax.set_axisbelow(True)
            
            ax.set_xlabel("Contexts", labelpad=8)
            ax.set_xticklabels(contexts, rotation=45, ha="right")
            
            # Leyenda indicando a qué clase pertenece el perfil
            ax.legend(loc="upper right", frameon=True, edgecolor="black")
            
            sns.despine()
            plt.tight_layout()
            
            # 5. Guardado sistemático de archivos
            filename_base = f"analisis/context_profile_clase_{n_clase}_{scale_suffix}"
            plt.savefig(f"{filename_base}.pdf", format="pdf", dpi=300)
            plt.savefig(f"{filename_base}.png", format="png", dpi=300)
            
            # Si estás ejecutando en bucle, puedes descomentar la siguiente línea 
            # para verlos todos en pantalla, o dejar que solo se guarden.
            # plt.show()
            plt.close(fig) # Cierra la figura actual para liberar memoria en el bucle


def plot_context_grid_analysis(context_model_analysis_df):
    """
    Genera dos figuras compuestas (mosaicos verticales):
    1. Una con escala lineal para cada n_clases.
    2. Otra con escala semilogarítmica para cada n_clases.
    """
    # 1. Identificar y ordenar las clases únicas disponibles
    unique_classes = sorted(context_model_analysis_df["n_clases"].unique())
    num_plots = len(unique_classes)
    
    if num_plots == 0:
        print("No hay datos en 'n_clases' para graficar.")
        return

    # Evaluamos ambas escalas por separado
    for scale in ["linear", "log"]:
        # Ajustamos el tamaño dinámicamente según el número de clases (alto = 2.5 pulgadas por clase)
        fig, axes = plt.subplots(
            nrows=num_plots, 
            ncols=1, 
            figsize=(8.5, 2.5 * num_plots), 
            sharex=True
        )
        
        # Si solo hay 1 clase, 'axes' no es una lista, lo convertimos para poder iterar uniformemente
        if num_plots == 1:
            axes = [axes]
            
        sns.set_style("white")

        # 2. Iterar sobre cada clase y su respectivo panel (axis)
        for i, n_clase in enumerate(unique_classes):
            ax = axes[i]
            
            # Filtrar y ordenar datos
            df_clase = context_model_analysis_df[context_model_analysis_df["n_clases"] == n_clase]
            df_clase = df_clase.sort_values(by="contexto")
            
            # Aseguramos que los contextos se traten como strings/categorías para el espaciado
            contexts = df_clase["contexto"].astype(str).tolist()
            values = df_clase["total_decisions"].tolist()
            
            # Dibujar la línea con marcadores circulares pequeños
            ax.plot(
                contexts, 
                values, 
                marker='o', 
                markersize=3.5, 
                color="#4c72b0", 
                linewidth=1.3, 
                label=f"$n_{{clases}} = {n_clase}$"
            )
            
            # Sombreado bajo la línea
            ax.fill_between(contexts, values, color="#4c72b0", alpha=0.08)
            
            # Configuración de escala
            if scale == "log":
                ax.set_yscale("log")
                if i == num_plots // 2: # Coloca la etiqueta larga en el panel del medio para que no se repita
                    ax.set_ylabel("Total decision bits (log scale)", labelpad=10)
            else:
                if i == num_plots // 2:
                    ax.set_ylabel("Total decision bits (linear)", labelpad=10)
            
            # Cuadrícula sutil (tanto en X como en Y para ayudar a la lectura vertical)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
            ax.xaxis.grid(True, linestyle=':', alpha=0.3, color='#bbbbbb')
            ax.set_axisbelow(True)
            
            # Leyenda elegante dentro de cada panel
            ax.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=0.9)
            
            # Quitar bordes sobrantes
            sns.despine(ax=ax)

        # 3. Formatear el eje X compartido (solo se aplica al último panel automáticamente)
        axes[-1].set_xlabel("Contexts", labelpad=10)
        axes[-1].set_xticklabels(contexts, rotation=45, ha="right")
        
        # Ajustar los márgenes para que no se corten los textos
        plt.tight_layout()
        
        # 4. Guardar los archivos de la composición completa
        filename_base = f"analisis/contexts_grid_distribution_{scale}"
        plt.savefig(f"{filename_base}.pdf", format="pdf", dpi=300)
        plt.savefig(f"{filename_base}.png", format="png", dpi=300)
        
        plt.show()
        plt.close(fig)

def plot_context_grid_bars(context_model_analysis_df):
    """
    Genera dos figuras compuestas en formato de gráficos de barras independientes:
    1. Una con escala lineal para cada n_clases.
    2. Otra con escala semilogarítmica para cada n_clases.
    """
    # 1. Identificar y ordenar las clases únicas
    unique_classes = sorted(context_model_analysis_df["n_clases"].unique())
    num_plots = len(unique_classes)
    
    if num_plots == 0:
        print("No hay datos en 'n_clases' para graficar.")
        return

    for scale in ["linear", "log"]:
        # Ajustamos el tamaño (8.5 de ancho da buen espacio para que los números queden rectos)
        fig, axes = plt.subplots(
            nrows=num_plots, 
            ncols=1, 
            figsize=(9.0, 2.6 * num_plots), 
            sharex=True
        )
        
        if num_plots == 1:
            axes = [axes]
            
        sns.set_style("white")

        # 2. Iterar por cada clase
        for i, n_clase in enumerate(unique_classes):
            ax = axes[i]
            
            df_clase = context_model_analysis_df[context_model_analysis_df["n_clases"] == n_clase]
            df_clase = df_clase.sort_values(by="contexto")
            
            contexts = df_clase["contexto"].astype(str).tolist()
            values = df_clase["total_decisions"].tolist()
            
            # Dibujar gráfico de barras en lugar de líneas
            ax.bar(
                contexts, 
                values, 
                color="#4c72b0", 
                edgecolor="black", 
                linewidth=0.6,
                width=0.7,  # Controla el ancho de la barra para dejar un espacio limpio entre ellas
                label=f"$n_{{clases}} = {n_clase}$"
            )
            
            # Configurar la escala del eje Y
            if scale == "log":
                ax.set_yscale("log")
                # Si es logarítmico, definimos un límite inferior pequeño (ej. 1) 
                # para evitar problemas visuales con barras que valgan cero.
                ax.set_ylim(bottom=1)
                if i == num_plots // 2:
                    ax.set_ylabel("Total decision bits (log scale)", labelpad=10)
            else:
                if i == num_plots // 2:
                    ax.set_ylabel("Total decision bits (linear)", labelpad=10)
            
            # Rejilla horizontal sutil (para barras, la rejilla vertical suele ser redundante)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
            ax.set_axisbelow(True)
            
            # Leyenda académica
            ax.legend(loc="upper right", frameon=True, edgecolor="black", framealpha=0.9)
            
            sns.despine(ax=ax)

        # 3. Formatear el eje X (solo en el último panel gracias a sharex=True)
        axes[-1].set_xlabel("Contexts", labelpad=10)
        
        # CAMBIO CLAVE: Cambiamos la rotación a 0 grados (horizontal) o 30 si prefieres un toque leve.
        # Al darle un ancho de figura de 9.0, los números del 0 al 36 caben perfectamente de frente.
        axes[-1].set_xticklabels(contexts, rotation=0, ha="center")
        
        plt.tight_layout()
        
        # 4. Guardar archivos
        filename_base = f"analisis/contexts_grid_bars_{scale}"
        plt.savefig(f"{filename_base}.pdf", format="pdf", dpi=300)
        plt.savefig(f"{filename_base}.png", format="png", dpi=300)
        
        plt.show()
        plt.close(fig)