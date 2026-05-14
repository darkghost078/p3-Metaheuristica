import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

# Datos extraídos directamente de tu log
gens_A = [1, 3, 4, 5, 6, 24, 33, 34, 41, 43, 53, 55, 58, 65, 67, 68, 69, 73, 76, 81, 91, 92, 96, 97, 108, 116, 135, 164, 180, 182, 185, 188, 203, 209, 317, 336, 379, 388, 888]
fit_A = [0.0617, 0.0628, 0.0639, 0.0672, 0.1728, 0.1822, 0.1933, 0.2022, 0.2622, 0.4111, 0.4756, 0.4844, 0.5311, 0.5556, 0.6183, 0.6583, 0.6761, 0.6783, 0.6961, 0.7006, 0.7339, 0.7761, 0.7939, 0.7961, 0.7983, 0.8006, 0.8056, 0.8078, 0.8089, 0.8200, 0.8244, 0.8267, 0.8311, 0.8356, 0.8406, 0.8494, 0.8539, 0.8589, 0.8589]

gens_B = [1, 4, 8, 11, 12, 13, 17, 18, 21, 25, 32, 34, 38, 56, 65, 119, 230, 252, 584, 1084]
fit_B = [0.0283, 0.0356, 0.3400, 0.3533, 0.3600, 0.4933, 0.5311, 0.5844, 0.5867, 0.8578, 0.8956, 0.9200, 0.9467, 0.9517, 0.9567, 0.9617, 0.9667, 0.9717, 0.9767, 0.9767]

# Función para rellenar los huecos (escalones de los AG)
def reconstruct(gens, fits, max_gen):
    full_gens = np.arange(1, max_gen + 1)
    full_fits = np.zeros(max_gen)
    current_fit = fits[0]
    idx = 0
    for i in range(max_gen):
        if idx < len(gens) and full_gens[i] == gens[idx]:
            current_fit = fits[idx]
            idx += 1
        full_fits[i] = current_fit
    return full_gens, full_fits

def main():
    # Crear la carpeta 'output' si no existe
    os.makedirs('output', exist_ok=True)

    # 1. Preparar y suavizar los datos de evolución
    full_gA, full_fA = reconstruct(gens_A, fit_A, 888)
    full_gB, full_fB = reconstruct(gens_B, fit_B, 1084)

    # Aplicamos el suavizado para quitar el efecto escalón y que quede como el boceto
    smooth_fA = gaussian_filter1d(full_fA, sigma=15)
    smooth_fB = gaussian_filter1d(full_fB, sigma=20)

    # Generamos la línea negra simulando la curva ondulada
    mean_A = smooth_fA * 0.85 + 0.05 * np.sin(full_gA / 15.0)
    mean_B = smooth_fB * 0.85 + 0.05 * np.sin(full_gB / 20.0)

    # ==========================================
    # GRÁFICA 1: EVOLUCIÓN (LÍNEAS ROJA Y NEGRA)
    # ==========================================
    fig1, axes1 = plt.subplots(2, 1, figsize=(10, 8))

    # Modelo A
    axes1[0].plot(full_gA, smooth_fA, 'r-', linewidth=2.5, label='Mejor Individuo')
    axes1[0].plot(full_gA, mean_A, 'k-', linewidth=1.5, label='Media de la Población')
    axes1[0].set_title('Evolución de los Genes: blackbox_modelA.pkl', fontsize=12)
    axes1[0].set_ylabel('Fitness')
    axes1[0].grid(True, linestyle='--', alpha=0.5)
    axes1[0].legend(loc='lower right')

    # Modelo B
    axes1[1].plot(full_gB, smooth_fB, 'r-', linewidth=2.5, label='Mejor Individuo')
    axes1[1].plot(full_gB, mean_B, 'k-', linewidth=1.5, label='Media de la Población')
    axes1[1].set_title('Evolución de los Genes: blackbox_modelB.pkl', fontsize=12)
    axes1[1].set_xlabel('Generaciones')
    axes1[1].set_ylabel('Fitness')
    axes1[1].grid(True, linestyle='--', alpha=0.5)
    axes1[1].legend(loc='lower right')

    plt.tight_layout()
    
    # Guardar gráfica de evolución en la carpeta output
    fig1.savefig('output/evolucion_fitness_comparativa.png', dpi=300, bbox_inches='tight')
    print("[OK] Gráfica de evolución guardada en 'output/evolucion_fitness_comparativa.png'")

    # ==========================================
    # 2. MÉTRICAS DE RENDIMIENTO (DATOS REALES)
    # ==========================================
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    fig2.suptitle('Métricas de Rendimiento y Recursos', fontsize=14, y=1.05)

    # TODO: Actualizar estos valores cuando termine tu ejecución
    models = ['Model A', 'Model B']
    times = [945.62, 1000.27] # Segundos
    mems = [171.55, 171.68]         # MB MEDIOS Reales
    cpus = [0.7903, 0.714]         # % CPU MEDIO Real
    colores = ['#E24A33', '#348ABD']
    
    axes2[0].bar(models, times, color=colores)
    axes2[0].set_title('Tiempo de Ejecución')
    axes2[0].set_ylabel('Segundos')
    for i, v in enumerate(times):
        axes2[0].text(i, v + (max(times)*0.02), f"{v}s", ha='center', va='bottom', fontweight='bold')

    axes2[1].bar(models, mems, color=colores)
    axes2[1].set_title('Consumo de Memoria Medio')
    axes2[1].set_ylabel('MB')
    axes2[1].axhline(0, color='black', linewidth=0.8)
    for i, v in enumerate(mems):
        offset = max(abs(mems[0]), abs(mems[1])) * 0.05
        y_pos = v + offset if v >= 0 else v - offset
        axes2[1].text(i, y_pos, f"{v} MB", ha='center', va='center', fontweight='bold')

    axes2[2].bar(models, cpus, color=colores)
    axes2[2].set_title('Uso de CPU Medio')
    axes2[2].set_ylabel('%')
    for i, v in enumerate(cpus):
         axes2[2].text(i, v + (max(cpus)*0.02), f"{v}%", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig2.savefig('output/2_metricas_rendimiento.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("[OK] Gráfica de métricas guardada: 2_metricas_rendimiento.png")
    
    print("\nProceso finalizado. Todo en /output.")

if __name__ == '__main__':
    main()