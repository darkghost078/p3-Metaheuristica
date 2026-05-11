import multiprocessing
import random
import copy
import joblib
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
)

warnings.filterwarnings("ignore")


# ==========================================
# UTILIDADES PARA ÁRBOLES
# ==========================================
def obtener_nodos(nodo):
    """Devuelve una lista con todos los nodos del árbol para poder elegir uno al azar."""
    nodos = [nodo]
    if hasattr(nodo, "hijo"):
        nodos.extend(obtener_nodos(nodo.hijo))
    if hasattr(nodo, "izq"):
        nodos.extend(obtener_nodos(nodo.izq))
    if hasattr(nodo, "der"):
        nodos.extend(obtener_nodos(nodo.der))
    return nodos


def reemplazar_nodo(arbol, nodo_viejo, nodo_nuevo):
    """Reemplaza recursivamente la primera instancia de nodo_viejo por nodo_nuevo."""
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
    """Cruza dos árboles intercambiando subárboles elegidos al azar."""
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
    """Muta un árbol reemplazando un nodo al azar por un nuevo subárbol generado aleatoriamente."""
    arbol_mutado = copy.deepcopy(arbol)
    nodos = obtener_nodos(arbol_mutado)

    if nodos:
        nodo_a_mutar = random.choice(nodos)
        nuevo_subarbol = generar_arbol_aleatorio(profundidad_max)
        arbol_mutado = reemplazar_nodo(arbol_mutado, nodo_a_mutar, nuevo_subarbol)

    return arbol_mutado


# ==========================================
# EVALUACIÓN (FITNESS) PARA MULTIPROCESAMIENTO
# ==========================================


def score_wrapper(params):
    return score(*params)


def score(arbol, p, bb):
    p_opt = encontrar_punto_mas_cercano(arbol, p)

    if p_opt is None:
        return 0.0

    num_puntos_esperados = 100
    distancias_prueba = [0.2, 0.4, 0.6]

    points = generar_vecindad_frontera(
        arbol, p_opt, num_puntos=num_puntos_esperados, paso=0.15
    )
    fitnessPoints = generar_puntos_ortogonales(
        arbol, points, distancias=distancias_prueba
    )

    wellClassified = 0

    total_evaluaciones_esperadas = num_puntos_esperados * len(distancias_prueba)

    if total_evaluaciones_esperadas == 0:
        return 0.0

    for point in fitnessPoints:
        predictMod1 = bb.predict([point[0]])[0]
        predictMod2 = bb.predict([point[1]])[0]

        try:
            predict1 = arbol.evaluar(point[0][0], point[0][1])
            predict2 = arbol.evaluar(point[1][0], point[1][1])
        except Exception:
            continue

        predict1 = 1 if predict1 > 0 else 0
        predict2 = 1 if predict2 > 0 else 0

        if predict1 == predictMod1 and predict2 == predictMod2 and predict1 != predict2:
            wellClassified += 1

    accuracy = wellClassified / total_evaluaciones_esperadas

    height = altura_arbol(arbol)
    coef_penal = 0.005

    fitness_final = accuracy - (coef_penal * height)
    return max(0.0, fitness_final)


# ==========================================
# CLASE MODELO CAJA NEGRA
# ==========================================
class BlackBoxModel:
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)

    def predict(self, X):
        return self.model.predict(X)


def inicializar_poblacion(tam_poblacion, profundidad_inicial):
    """Inicializa la población de árboles con individuos aleatorios y calcula su fitness."""
    poblacion = []
    for _ in range(tam_poblacion):
        arbol = generar_arbol_aleatorio(profundidad_inicial)
        poblacion.append((arbol, 0.0))  # El fitness inicial es 0.0
    return poblacion


def evaluar_poblacion(poblacion, p, bb):
    """Evalúa toda la población usando multiprocesamiento."""
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        params = [(individuo, p, bb) for individuo, _ in poblacion]
        resultados_fitness = pool.map(score_wrapper, params)

    # Actualizamos la población con el fitness calculado (vector por pares: individuo, fitness)
    poblacion_evaluada = [
        (poblacion[i][0], resultados_fitness[i]) for i in range(len(poblacion))
    ]

    # Ordenamos de mayor a menor fitness
    poblacion_evaluada.sort(key=lambda x: x[1], reverse=True)
    return poblacion_evaluada


def seleccion_torneo(poblacion, k=3):
    """Selección por torneo para elegir padres."""
    seleccionados = random.sample(poblacion, k)
    seleccionados.sort(key=lambda x: x[1], reverse=True)
    return seleccionados[0][0]  # Retorna el mejor individuo (árbol)


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
    print(f"\n--- Generando gráfica de fitness para el punto origen {punto_origen} ---")
    punto_central = encontrar_punto_mas_cercano(arbol, punto_origen)

    if not punto_central:
        print(
            "El optimizador falló o devolvió nulo. No hay frontera útil para graficar."
        )
        return

    puntos_frontera = generar_vecindad_frontera(
        arbol, punto_central, num_puntos=100, paso=paso
    )
    vector_ortogonales = generar_puntos_ortogonales(
        arbol, puntos_frontera, distancias=distancias
    )

    plt.figure(figsize=(10, 10))

    X_front = [p[0] for p in puntos_frontera]
    Y_front = [p[1] for p in puntos_frontera]
    plt.plot(X_front, Y_front, "b-", label="Frontera f(x,y)=0", linewidth=2)

    colores = ["r", "g", "orange"]

    # Extraemos los puntos leyendo la lista plana con saltos proporcionales
    for idx_distancia, d in enumerate(distancias):
        x_mas, y_mas = [], []
        x_menos, y_menos = [], []

        for i in range(idx_distancia, len(vector_ortogonales), len(distancias)):
            p_m, p_me = vector_ortogonales[i]
            x_mas.append(p_m[0])
            y_mas.append(p_m[1])
            x_menos.append(p_me[0])
            y_menos.append(p_me[1])

        plt.scatter(
            x_mas,
            y_mas,
            color=colores[idx_distancia],
            s=10,
            alpha=0.6,
            label=f"Distancia +{d}",
        )
        plt.scatter(
            x_menos,
            y_menos,
            color=colores[idx_distancia],
            s=10,
            alpha=0.6,
            marker="x",
            label=f"Distancia -{d}",
        )

    plt.plot(
        punto_origen[0], punto_origen[1], "k*", markersize=12, label="Punto Origen"
    )

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(color="gray", linestyle="--", linewidth=0.2)
    plt.legend()
    plt.title("Puntos de Evaluación de Fitness del Mejor Individuo")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.axis("equal")


# ==========================================
# ALGORITMO GENÉTICO
# ==========================================
def genetico(
    path_modelo,
    tam_poblacion=20,
    tam_elite=5,
    generaciones=50,
    prob_cruce=0.8,
    prob_mutacion=0.2,
    profundidad_inicial=4,
):
    # Generar dataset de entrenamiento con puntos aleatorios usando el modelo caja negra
    print(f"Generando dataset exploratorio para {path_modelo}...")
    bb = BlackBoxModel(path_modelo)

    # Generar puntos de manera más distribuida para encontrar ambas clases
    p = puntos(bb)[0]

    print("\n=== INICIANDO ALGORITMO GENÉTICO ===")
    # 1. Inicializar
    poblacion = inicializar_poblacion(
        tam_poblacion=tam_poblacion, profundidad_inicial=profundidad_inicial
    )

    # 2. Bucle Generacional
    for gen in range(generaciones):
        # Evaluar
        poblacion = evaluar_poblacion(poblacion, p, bb)

        mejor_fitness = poblacion[0][1]
        print(
            f"Generación {gen + 1}/{generaciones} | Mejor Fitness (Accuracy): {mejor_fitness:.4f}"
        )

        # Si encontramos el modelo perfecto
        if mejor_fitness >= 0.99:
            print("¡Solución óptima alcanzada!")
            break

        nueva_poblacion = []

        # 3. Elitismo (Guardar los mejores)
        elite = poblacion[:tam_elite]
        nueva_poblacion.extend(
            [(ind, 0.0) for ind, _ in elite]
        )  # Reseteamos fitness para la sig generacion

        # 4. Cruzamiento y Mutación para rellenar la población
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

    # Evaluación final
    poblacion = evaluar_poblacion(poblacion, p, bb)
    mejor_individuo = poblacion[0]
    print("\n=== FIN DEL ALGORITMO ===")
    print(f"Mejor Fitness Final: {mejor_individuo[1]:.4f}")
    print(f"Ecuación Encontrada para la Frontera:\n{mejor_individuo[0]}")
    return mejor_individuo[0], p


if __name__ == "__main__":
    # Ejecutamos el genético para el modelo A
    print("\n----------------------------------------------------")
    print("EJECUTANDO GENÉTICO PARA: blackbox_modelA.pkl")
    print("----------------------------------------------------")
    # Los parámetros: 20 individuos, 5 élite según las especificaciones
    ag_A, point_A = genetico(
        "blackbox_modelA.pkl", tam_poblacion=20, tam_elite=2, generaciones=20
    )
    ag_A.graf()
    graficar_fitness_resultado(ag_A, point_A, distancias=[0.2, 0.4, 0.6], paso=0.15)
    plt.show()
    # Ejecutamos el genético para el modelo B
    print("\n----------------------------------------------------")
    print("EJECUTANDO GENÉTICO PARA: blackbox_modelB.pkl")
    print("----------------------------------------------------")
    ag_B, point_B = genetico(
        "blackbox_modelB.pkl", tam_poblacion=20, tam_elite=2, generaciones=20
    )
    ag_B.graf()
    graficar_fitness_resultado(ag_B, point_B, distancias=[0.2, 0.4, 0.6], paso=0.15)
    plt.show()
