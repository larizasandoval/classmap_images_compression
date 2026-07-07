"""
Script para el análisis de resultados. Genera gráficos y tablas del pipeline

"""

from pipeline2 import pipeline, residual_stats,context_model_analysis
from sintetic_img_generator import generate_synthetic_map
import os
import numpy as np
import pandas as pd
from plots import *
import re

#----------------------------
# Main script para ejecutar el análisis de resultados, generar tablas y gráficos
#----------------------------


if __name__ == "__main__":
    
    # Ruta base donde se encuentran todos los sensores
    base_path = "images_complete/green_book_corpus"
    
    # Listar todas las carpetas de sensores que existen en la ruta base
    # Filtrado para asegurarse de que solo lea directorios
    sensors = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    #sensors = ['airs']  # Lista de sensores a procesar
    #Dataframes para almacenar los resultados

    entropy_results = [] # Para almacenar los resultados de entropía y tasa de bits de cada imagen
    resisual_stats_results = [] # Para almacenar los resultados de estadísticas de residuos
    context_model_analysis_results = [] # Para almacenar los resultados del análisis del modelo de contexto

    # Bucle para recorrer cada sensor dinámicamente
    for sensor_name in sensors:
        path_maps = os.path.join(base_path, sensor_name, "maps")
        
        # Validar si la carpeta "maps" existe para este sensor antes de continuar
        if not os.path.exists(path_maps):
            continue
            
        img_names = os.listdir(path_maps)
        
        #Aplicar el pipeline a cada imagen y almacenar los resultados
        
        for img_name, i in zip(img_names, range(0, len(img_names))):
            #bloque_dimensiones = [p for p in img_name.split('-') if re.search(r'\d+x\d+x\d+', p)][-1]
            #print(f"prubea: {img_name} -> {type(bloque_dimensiones)}")
            #break
            try:
                # 1. Separamos por guiones bajos para aislar el bloque dinámico con 'x' (ej: "6x1024x1024")
                bloque_dimensiones = [p for p in img_name.split('-') if re.search(r'\d+x\d+x\d+', p)][0]

                # 2. Separamos el bloque por las 'x' -> ['6', '1024', '1024']
                dimensiones = bloque_dimensiones.split('x')
                
                # 3. Aplicamos tu regla Z-Y-X: ignoramos dimensiones[0] y tomamos las otras dos
                filas = int(dimensiones[1])     # y
                columnas = int(dimensiones[2].replace('.raw', ''))  # x
                
                # 4. Cortar el nombre justo donde termina el bloque de dimensiones
                #indice_corte = img_name.find(bloque_dimensiones) + len(bloque_dimensiones)
                map_name_clean = img_name#img_name[:indice_corte]
                
            except Exception as e:
                # Valores por defecto de respaldo por si acaso
                filas, columnas = 1024, 1024
                map_name_clean = os.path.splitext(img_name)[0]
                print(f"Advertencia: No se pudo procesar el nombre o ZYX en {img_name}. Usando {filas}x{columnas}. Error: {e}")

            try:
                # Leer la imagen binaria completa y aplicar el reshape directo con Y y X
                img = np.fromfile(os.path.join(path_maps, img_name), dtype=np.uint8).reshape((filas, columnas))
                
            except Exception as e:
                print(f"Error crítico al leer o redimensionar la imagen {img_name}: {e}")
                continue

            total_pixeles = int(img.size) # Total de píxeles (filas x columnas)
            
            data = pipeline(img)
            data["sensor"] = sensor_name
            data["map"] = map_name_clean
            data["total_pixeles"] = total_pixeles
            entropy_results.append(data)

            data2 = residual_stats(img)
            data2["sensor"] = sensor_name
            data2["map"] = map_name_clean
            data2["n_clases"] = int(img.max() + 1)
            data2["total_pixeles"] = total_pixeles
            resisual_stats_results.append(data2)
            
            data3 = context_model_analysis(img)
            data3["sensor"] = sensor_name
            data3["map"] = map_name_clean
            data3["total_pixeles"] = total_pixeles
            context_model_analysis_results.append(data3)
        
    #----------------------------------
    #--- ----Guardar resultados (Append)---------
    #--------------------------------
    datos_paper = {
    "n_clases": [4, 7, 9, 17, 32],
    "bbp_paper": [0.1764, 0.2793, 0.4869, 0.9146, 2.4147]
    }

    df_paper = pd.DataFrame(datos_paper)
    
    # --- Guardar Resultados de Entropía sin sobrescribir ---
    entropy_file = "analisis2//entropy_results.xlsx"
    entropy_results_df = pd.DataFrame(entropy_results)
    
    if os.path.exists(entropy_file):
        df_existente_ent = pd.read_excel(entropy_file)
        entropy_results_df = pd.concat([df_existente_ent, entropy_results_df], ignore_index=True)
        
    entropy_results_df = entropy_results_df.sort_values(by="n_clases").reset_index(drop=True)
    entropy_results_df.to_excel(entropy_file, index=False)
    
    #print(entropy_results_df)
    '''entropy_results_df = pd.merge(entropy_results_df, df_paper, on="n_clases", how="left")
    entropy_results_df['diff_bpp'] = (entropy_results_df['bpp_final'] - entropy_results_df['bbp_paper']) / entropy_results_df['bbp_paper'] * 100
    entropy_results_df.to_excel( "analisis/entropy_results_paper_diff.xlsx",
                                 columns=["n_clases", "bbp_paper", "bpp_final", "diff_bpp"],
                                 index=False)'''
    
    
    
    # --- Guardar Estadísticas de Residuos sin sobrescribir ---
    residual_file = "analisis/residual_stats_results.xlsx"
    residual_stats_df = pd.DataFrame(resisual_stats_results)
    
    if os.path.exists(residual_file):
        df_existente_res = pd.read_excel(residual_file)
        residual_stats_df = pd.concat([df_existente_res, residual_stats_df], ignore_index=True)
        
    residual_stats_df = residual_stats_df.sort_values(by="n_clases").reset_index(drop=True)
    residual_stats_df.to_excel(residual_file, index=False)
    
    
    #-----Context model analysis results
    #temp = [pd.DataFrame(res) for res in context_model_analysis_results]
    #context_model_analysis_df = pd.concat(temp, ignore_index=True).sort_values(by="n_clases").reset_index(drop=True)
    #context_model_analysis_df.to_excel("analisis/context_model_analysis_results.xlsx", index=False)
    
    #---------------
    #-----Plots-----
    #---------------

    #plot_entropy_comparison(entropy_results_df,details=False)

    #plot_entropy_comparison(entropy_results_df,details=True)

    #plot_residual_stats(residual_stats_df)

    #plot_context_model_analysis(context_model_analysis_df)

    #plot_context_grid_bars(context_model_analysis_df)