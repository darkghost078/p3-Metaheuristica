import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

def main():
    # Crear la carpeta 'output' si no existe
    os.makedirs('output', exist_ok=True)

    # ==========================================
    # 1. CARGAR DATOS DEL MODELO A (REALES DESDE CSV)
    # ==========================================
    try:
        df_A = pd.read_csv('resultadosMA.csv')
        full_gA = df_A['gen'].values
        smooth_fA = gaussian_filter1d(df_A['mejorfitness'].values, sigma=15)
        col_media_A = 'mediafitness' if 'mediafitness' in df_A.columns else 'mediapoblacion'
        mean_A = gaussian_filter1d(df_A[col_media_A].values, sigma=3)
    except Exception as e:
        print(f"Error al leer resultadosMA.csv: {e}")
        return

    # ==========================================
    # 2. CARGAR DATOS DEL MODELO B (REALES DESDE CSV)
    # ==========================================
    try:
        df_B = pd.read_csv('resultadosMB.csv')
        full_gB = df_B['gen'].values
        smooth_fB = gaussian_filter1d(df_B['mejorfitness'].values, sigma=15)
        col_media_B = 'mediafitness' if 'mediafitness' in df_B.columns else 'mediapoblacion'
        mean_B = gaussian_filter1d(df_B[col_media_B].values, sigma=3)
    except Exception as e:
        print(f"Error al leer resultadosMB.csv: {e}")
        return

    # ==========================================
    # GRÁFICA 1: EVOLUCIÓN (LÍNEAS ROJA Y NEGRA)
    # ==========================================
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))

    # Panel del Modelo A
    axes1[0].plot(full_gA, smooth_fA, 'r-', linewidth=2.5, label='Mejor Individuo')
    axes1[0].plot(full_gA, mean_A, 'k-', linewidth=1.2, alpha=0.85, label='Media de la Población')
    axes1[0].set_title('Evolución de los Genes: blackbox_modelA.pkl', fontsize=12)
    axes1[0].set_ylabel('Fitness')
    axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[0].legend(loc='lower right')

    # Panel del Modelo B 
    axes1[1].plot(full_gB, smooth_fB, 'r-', linewidth=2.5, label='Mejor Individuo')
    axes1[1].plot(full_gB, mean_B, 'k-', linewidth=1.2, alpha=0.85, label='Media de la Población')
    axes1[1].set_title('Evolución de los Genes: blackbox_modelB.pkl', fontsize=12)
    axes1[1].set_xlabel('Generaciones')
    axes1[1].set_ylabel('Fitness')
    axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[1].legend(loc='lower right')

    plt.tight_layout()
    fig1.savefig('output/1_evolucion_fitness_comparativa.png', dpi=300, bbox_inches='tight')
    print("[OK] Gráfica de evolución guardada en 'output/1_evolucion_fitness_comparativa.png'")

    # ==========================================
    # GRÁFICA 2: MÉTRICAS DE RENDIMIENTO
    # ==========================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle('Métricas de Rendimiento y Recursos', fontsize=14, y=1.05)

    models = ['Model A', 'Model B']
    times = [3159.15, 2940.69]
    mems = [184.88, 186.02]
    cpus = [86.57, 87.43]
    colores = ['#E24A33', '#348ABD']
    
    # Panel Tiempos
    axes2[0].bar(models, times, color=colores)
    axes2[0].set_title('Tiempo de Ejecución')
    axes2[0].set_ylabel('Segundos')
    for i, v in enumerate(times):
        axes2[0].text(i, v + (max(times)*0.02), f"{v}s", ha='center', va='bottom', fontweight='bold')

    # Panel Memoria
    axes2[1].bar(models, mems, color=colores)
    axes2[1].set_title('Consumo de Memoria Medio')
    axes2[1].set_ylabel('MB')
    axes2[1].axhline(0, color='black', linewidth=0.8)
    for i, v in enumerate(mems):
        offset = max(abs(mems[0]), abs(mems[1])) * 0.05
        y_pos = v + offset if v >= 0 else v - offset
        axes2[1].text(i, y_pos, f"{v} MB", ha='center', va='center', fontweight='bold')

    # Panel CPU
    axes2[2].bar(models, cpus, color=colores)
    axes2[2].set_title('Uso de CPU Medio')
    axes2[2].set_ylabel('%')
    for i, v in enumerate(cpus):
         axes2[2].text(i, v + (max(cpus)*0.02), f"{v}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig2.savefig('output/2_metricas_rendimiento.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("[OK] Gráfica de métricas guardada: 2_metricas_rendimiento.png")
    
    print("\nProceso finalizado. Todo guardado en la carpeta /output.")

if __name__ == '__main__':
    main()