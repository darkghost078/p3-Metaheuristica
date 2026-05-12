import multiprocessing
import random
import copy
import joblib
import time
import psutil
import os
import numpy as np
import warnings
import matplotlib.pyplot as plt

from arbol import (
    generar_arbol_aleatorio,
    encontrar_punto_mas_cercano,
    generar_puntos_ortogonales,
    generar_vecindad_frontera,
    altura_arbol,
    simplificar,
    encontrar_intersecciones_rayos,
)

warnings.filterwarnings("ignore")


def obtener_nodos(nodo):
    nodos = [nodo]
    if hasattr(nodo, "hijo"):
        nodos.extend(obtener_nodos(nodo.hijo))
    if hasattr(nodo, "izq"):
        nodos.extend(obtener_nodos(nodo.izq))
    if hasattr(nodo, "der"):
        nodos.extend(obtener_nodos(nodo.der))
    return nodos


def reemplazar_nodo(arbol, nodo_viejo, nodo_nuevo):
    if arbol is nodo_viejo:
        return copy.deepcopy(nodo_nuevo)

    if hasattr(arbol, "hijo"):
        if arbol.hijo is nodo_viejo:
            arbol.hijo = copy.deepcopy(nodo_nuevo)
        else:
            arbol.hijo = reemplazar_nodo(arbol.hijo, nodo_viejo, nodo_nuevo)

    if hasattr(arbol, "izq"):
        if arbol.izq is nodo_viejo:
            arbol.izq = copy.deepcopy(nodo_nuevo)
        else:
            arbol.izq = reemplazar_nodo(arbol.izq, nodo_viejo, nodo_nuevo)

    if hasattr(arbol, "der"):
        if arbol.der is nodo_viejo:
            arbol.der = copy.deepcopy(nodo_nuevo)
        else:
            arbol.der = reemplazar_nodo(arbol.der, nodo_viejo, nodo_nuevo)

    return arbol


def cruzar_arboles(padre1, padre2):
    hijo1 = copy.deepcopy(padre1)
    hijo2 = copy.deepcopy(padre2)

    nodos1 = obtener_nodos(hijo1)
    nodos2 = obtener_nodos(hijo2)

    if nodos1 and nodos2:
        n1 = random.choice(nodos1)
        n2 = random.choice(nodos2)
        hijo1 = reemplazar_nodo(hijo1, n1, n2)
        hijo2 = reemplazar_nodo(hijo2, n2, n1)

    return hijo1, hijo2


def mutar_arbol(arbol, profundidad_max=2):
    arbol_mutado = copy.deepcopy(arbol)
    nodos = obtener_nodos(arbol_mutado)

    if nodos:
        nodo_a_mutar = random.choice(nodos)
        nuevo_subarbol = generar_arbol_aleatorio(profundidad_max)
        arbol_mutado = reemplazar_nodo(arbol_mutado, nodo_a_mutar, nuevo_subarbol)

    return arbol_mutado


def score_wrapper(params):
    return score(*params)


def score(arbol, p, bb):
    centros_brutos = encontrar_intersecciones_rayos(
        arbol, p, num_rayos=8, radio_max=10.0, paso=0.3
    )

    if not centros_brutos:
        return 0.0

    # Purgar centros que convergen en el mismo espacio físico
    centros = []
    for c in centros_brutos:
        if not any(np.hypot(c[0] - cv[0], c[1] - cv[1]) < 0.15 for cv in centros):
            centros.append(c)

    # Si la curva es un micropunto colapsado, el árbol no es viable
    if len(centros) < 3:
        return 0.0

    # Penalización por asfixia si no hay margen de separación real
    distancias_origen = [np.hypot(c[0] - p[0], c[1] - p[1]) for c in centros]
    distancia_media = sum(distancias_origen) / len(distancias_origen)

    penalizacion_margen = 0.0
    if distancia_media < 0.3:
        penalizacion_margen = 0.2

    num_puntos_por_centro = 10
    distancias_prueba = [0.2, 0.4, 0.6]

    wellClassified = 0
    fallos_matematicos = 0
    fallos_gradiente = 0
    total_evaluaciones = 0

    for centro in centros:
        points = generar_vecindad_frontera(
            arbol, centro, num_puntos=num_puntos_por_centro, paso=0.15
        )

        fitnessPoints, fallos_grad = generar_puntos_ortogonales(
            arbol, points, distancias=distancias_prueba
        )

        fallos_gradiente += fallos_grad
        total_evaluaciones += len(points) * len(distancias_prueba)

        for point in fitnessPoints:
            predictMod1 = bb.predict([point[0]])[0]
            predictMod2 = bb.predict([point[1]])[0]

            try:
                predict1 = arbol.evaluar(point[0][0], point[0][1])
                predict2 = arbol.evaluar(point[1][0], point[1][1])

                if abs(predict1) > 1000 or abs(predict2) > 1000:
                    fallos_matematicos += 1
                    continue
            except Exception:
                fallos_matematicos += 1
                continue

            predict1_bin = 1 if predict1 > 0 else 0
            predict2_bin = 1 if predict2 > 0 else 0

            if (
                predict1_bin == predictMod1
                and predict2_bin == predictMod2
                and predict1_bin != predict2_bin
            ):
                wellClassified += 1

    if total_evaluaciones == 0:
        return 0.0

    ratio_fallos = (fallos_matematicos + fallos_gradiente) / total_evaluaciones

    if ratio_fallos > 0.20:
        return 0.0

    accuracy = wellClassified / total_evaluaciones
    height = altura_arbol(arbol)
    coef_penal = 0.005

    # Aplicamos la reducción si no cumple el estándar geométrico
    fitness_final = accuracy - (coef_penal * height) - penalizacion_margen

    return max(0.0, fitness_final)


class BlackBoxModel:
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, X):
        return self.model.predict(X)


def inicializar_poblacion(tam_poblacion, profundidad_inicial):
    poblacion = []
    for _ in range(tam_poblacion):
        arbol = generar_arbol_aleatorio(profundidad_inicial)
        poblacion.append((arbol, 0.0))
    return poblacion


def evaluar_poblacion(poblacion, p, bb):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        params = [(individuo, p, bb) for individuo, _ in poblacion]
        resultados_fitness = pool.map(score_wrapper, params)

    poblacion_evaluada = [
        (poblacion[i][0], resultados_fitness[i]) for i in range(len(poblacion))
    ]

    poblacion_evaluada.sort(key=lambda x: x[1], reverse=True)
    return poblacion_evaluada


def seleccion_torneo(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    seleccionados.sort(key=lambda x: x[1], reverse=True)
    return seleccionados[0][0]


def puntos(bb):
    print("\n=== Buscando puntos de ambas clases ===")
    class0 = []
    class1 = []
    paso = 0.25
    limite = paso

    while len(class0) == 0 or len(class1) == 0:
        values = np.arange(-limite, limite + paso, paso)
        for x in values:
            for y in values:
                punto = np.array([[round(x, 2), round(y, 2)]])
                prediccion = bb.predict(punto)[0]

                if prediccion == 0 and len(class0) < 50:
                    class0.append(punto[0])
                elif prediccion == 1 and len(class1) < 50:
                    class1.append(punto[0])

                if len(class0) > 0 and len(class1) > 0:
                    break
            if len(class0) > 0 and len(class1) > 0:
                break

        if len(class0) == 0 or len(class1) == 0:
            limite += paso
            if limite > 100:
                print("Límite de búsqueda excedido.")
                return class0 + class1

    if len(class0) < len(class1):
        return class0
    else:
        return class1


def graficar_fitness_resultado(
    arbol, punto_origen, distancias=[0.1, 0.2, 0.3], paso=0.05
):
    print(f"\n--- Generando gráfica de fitness para el origen {punto_origen} ---")
    centros = encontrar_intersecciones_rayos(
        arbol, punto_origen, num_rayos=8, radio_max=10.0, paso=0.5
    )

    if not centros:
        print(
            "Los rayos no intersectaron la frontera en el radio máximo. No se genera gráfica."
        )
        return

    plt.figure(figsize=(10, 10))
    colores = ["r", "g", "orange", "purple", "cyan", "brown"]

    for idx_centro, centro in enumerate(centros):
        puntos_frontera = generar_vecindad_frontera(
            arbol, centro, num_puntos=10, paso=paso
        )
        vector_ortogonales, _ = generar_puntos_ortogonales(
            arbol, puntos_frontera, distancias=distancias
        )

        X_front = [p[0] for p in puntos_frontera]
        Y_front = [p[1] for p in puntos_frontera]
        plt.plot(X_front, Y_front, "b-", linewidth=2, alpha=0.5)

        plt.plot(
            [punto_origen[0], centro[0]],
            [punto_origen[1], centro[1]],
            "k--",
            alpha=0.2,
            linewidth=0.8,
        )
        plt.plot(centro[0], centro[1], "mD", markersize=5)

        for idx_distancia, d in enumerate(distancias):
            x_mas, y_mas = [], []
            x_menos, y_menos = [], []

            for i in range(idx_distancia, len(vector_ortogonales), len(distancias)):
                p_m, p_me = vector_ortogonales[i]
                x_mas.append(p_m[0])
                y_mas.append(p_m[1])
                x_menos.append(p_me[0])
                y_menos.append(p_me[1])

            lbl_mas = f"Distancia +{d}" if idx_centro == 0 else ""
            lbl_menos = f"Distancia -{d}" if idx_centro == 0 else ""

            color_actual = colores[idx_distancia % len(colores)]
            plt.scatter(
                x_mas, y_mas, color=color_actual, s=10, alpha=0.6, label=lbl_mas
            )
            plt.scatter(
                x_menos,
                y_menos,
                color=color_actual,
                s=10,
                alpha=0.6,
                marker="x",
                label=lbl_menos,
            )

    plt.plot(
        punto_origen[0], punto_origen[1], "k*", markersize=12, label="Punto Origen"
    )

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(color="gray", linestyle="--", linewidth=0.2)
    plt.legend()
    plt.title("Evaluación Multidireccional Optimizada")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.axis("equal")


def genetico(
    path_modelo,
    tam_poblacion=20,
    tam_elite=5,
    generaciones=100,
    prob_cruce=0.8,
    prob_mutacion=0.2,
    profundidad_inicial=4,
):
    print(f"\n>>> Iniciando ejecución y test de rendimiento para: {path_modelo}")
    if not os.path.exists("output"):
        os.makedirs("output")
        print("Carpeta 'output' creada.")

    proceso = psutil.Process(os.getpid())
    mem_inicial = proceso.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    print(f"Generando dataset exploratorio para {path_modelo}...")
    bb = BlackBoxModel(path_modelo)
    p = puntos(bb)[0]

    print("\n=== INICIANDO ALGORITMO GENÉTICO ===")
    poblacion = inicializar_poblacion(
        tam_poblacion=tam_poblacion, profundidad_inicial=profundidad_inicial
    )

    for gen in range(generaciones):
        poblacion = evaluar_poblacion(poblacion, p, bb)
        mejor_fitness = poblacion[0][1]
        print(
            f"Generación {gen + 1}/{generaciones} | Mejor Fitness: {mejor_fitness:.4f}"
        )

        if mejor_fitness >= 0.99:
            print("¡Solución óptima alcanzada!")
            break

        nueva_poblacion = []
        elite = poblacion[:tam_elite]
        nueva_poblacion.extend([(ind, 0.0) for ind, _ in elite])

        while len(nueva_poblacion) < tam_poblacion:
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)

            if random.random() < prob_cruce:
                hijo1, hijo2 = cruzar_arboles(padre1, padre2)
            else:
                hijo1, hijo2 = copy.deepcopy(padre1), copy.deepcopy(padre2)

            if random.random() < prob_mutacion:
                hijo1 = mutar_arbol(hijo1)
            if random.random() < prob_mutacion:
                hijo2 = mutar_arbol(hijo2)

            hijo1 = simplificar(hijo1)
            hijo2 = simplificar(hijo2)

            nueva_poblacion.append((hijo1, 0.0))
            if len(nueva_poblacion) < tam_poblacion:
                nueva_poblacion.append((hijo2, 0.0))

        poblacion = nueva_poblacion

    poblacion = evaluar_poblacion(poblacion, p, bb)
    mejor_arbol = poblacion[0][0]
    mejor_fitness = poblacion[0][1]

    end_time = time.time()
    mem_final = proceso.memory_info().rss / (1024 * 1024)
    tiempo_total = end_time - start_time
    uso_memoria = mem_final - mem_inicial
    cpu_uso = psutil.cpu_percent(interval=1)

    print("\n=== FIN DEL ALGORITMO ===")
    print(f"--- RESULTADOS PARA {path_modelo} ---")
    print(f"Mejor Fitness Final: {mejor_fitness:.4f}")
    print(
        f"Tiempo Total: {tiempo_total:.2f} s | Memoria: {uso_memoria:.2f} MB | CPU: {cpu_uso}%"
    )
    print(f"Ecuación:\n{mejor_arbol}")

    nombre_base = os.path.basename(path_modelo).replace(".pkl", "")
    mejor_arbol.graf()
    plt.title(f"Frontera Generada - {nombre_base}")
    plt.savefig(f"output/frontera_{nombre_base}.png")
    plt.close()

    graficar_fitness_resultado(mejor_arbol, p, distancias=[0.2, 0.4, 0.6], paso=0.15)
    plt.title(f"Evaluación Fitness - {nombre_base}")
    plt.savefig(f"output/fitness_{nombre_base}.png")
    plt.close()

    print(
        f"Gráficas guardadas en la carpeta 'output/' como frontera_{nombre_base}.png y fitness_{nombre_base}.png"
    )

    return mejor_arbol, p


if __name__ == "__main__":
    modelos_a_evaluar = ["blackbox_modelA.pkl", "blackbox_modelB.pkl"]
    print("====================================================")
    print("INICIANDO EVALUACIÓN DE PRÁCTICA 3 - METAHEURÍSTICAS")
    print("====================================================")

    for modelo in modelos_a_evaluar:
        try:
            if not os.path.exists(modelo):
                print(f"\n[!] Error: No se encuentra el archivo {modelo}. Saltando...")
                continue

            arbol_final, punto_inicio = genetico(
                path_modelo=modelo, tam_poblacion=20, tam_elite=2, generaciones=1000
            )

            print(f"\n[OK] Finalizado éxito: {modelo}")
            print("-" * 50)

        except Exception as e:
            print(f"\n[!] Error crítico procesando {modelo}: {e}")

    print("\n====================================================")
    print("PROCESO COMPLETADO. Revisa la carpeta 'output' para las gráficas.")
    print("====================================================")

