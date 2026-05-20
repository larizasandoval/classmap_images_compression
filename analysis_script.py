
"""
Script para el análisis de resultados. Genera gráficos y tablas del pipeline

"""

from predictor import pipeline, residual_stats,context_model_analysis
from sintetic_img_generator import generate_synthetic_map
import os
import numpy as np
import pandas as pd
from plots import *

#----------------------------
# Main script para ejecutar el análisis de resultados, generar tablas y gráficos
#----------------------------


if __name__ == "__main__":
    
    # Cargar mapas de interes 
    img_names = os.listdir("curated_maps")
    imgs = [np.fromfile(os.path.join("curated_maps", name), dtype=np.uint8).reshape((512, 614)) 
            for name in img_names]

    entropy_results = [] # Para almacenar los resultados de entropía y tasa de bits de cada imagen
    resisual_stats_results = [] # Para almacenar los resultados de estadísticas de residuos
    context_model_analysis_results = [] # Para almacenar los resultados del análisis del modelo de contexto


    for img in imgs:
        data = pipeline(img)
        entropy_results.append(data)

        data2 = residual_stats(img)
        data2["n_clases"] = img.max() + 1
        resisual_stats_results.append(data2)
        
        data3 = context_model_analysis(img)
        context_model_analysis_results.append(data3)
    
    #----------------------------------
    #--- ----Entropy results---------
    #--------------------------------
    datos_paper = {
    "n_clases": [4, 7, 9, 17, 32],
    "bbp_paper": [0.1764, 0.2793, 0.4869, 0.9146, 2.4147]
    }

    df_paper = pd.DataFrame(datos_paper)
    entropy_results_df = pd.DataFrame(entropy_results).sort_values(by="n_clases").reset_index(drop=True)
    entropy_results_df.to_excel("analisis/entropy_results.xlsx", index=False)
    entropy_results_df = pd.merge(entropy_results_df, df_paper, on="n_clases", how="left")
    entropy_results_df['diff_bpp'] = (entropy_results_df['bpp_final'] - entropy_results_df['bbp_paper']) / entropy_results_df['bbp_paper'] * 100
    entropy_results_df.to_excel( "analisis/entropy_results_paper_diff.xlsx",
                                 columns=["n_clases", "bbp_paper", "bpp_final", "diff_bpp"],
                                 index=False)
    
    #-----Residual stats results
    residual_stats_df = pd.DataFrame(resisual_stats_results)
    residual_stats_df = residual_stats_df.sort_values(by="n_clases")
    residual_stats_df.to_excel("analisis/residual_stats_results.xlsx", index=False)
    #-----Context model analysis results
    temp = [pd.DataFrame(res) for res in context_model_analysis_results]
    context_model_analysis_df = pd.concat(temp, ignore_index=True).sort_values(by="n_clases").reset_index(drop=True)
    context_model_analysis_df.to_excel("analisis/context_model_analysis_results.xlsx", index=False)
    
    
    #-----Plots

    plot_entropy_comparison(entropy_results_df,details=False)

    plot_entropy_comparison(entropy_results_df,details=True)

    plot_residual_stats(residual_stats_df)

    plot_context_model_analysis(context_model_analysis_df)
    