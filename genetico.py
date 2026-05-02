import multiprocessing
import random
import copy
import joblib
import numpy as np
import warnings
from arbol import generar_arbol_aleatorio, Nodo

warnings.filterwarnings('ignore')

# ==========================================
# UTILIDADES PARA ÁRBOLES
# ==========================================
def obtener_nodos(nodo):
    """Devuelve una lista con todos los nodos del árbol para poder elegir uno al azar."""
    nodos = [nodo]
    if hasattr(nodo, 'hijo'):
        nodos.extend(obtener_nodos(nodo.hijo))
    if hasattr(nodo, 'izq'):
        nodos.extend(obtener_nodos(nodo.izq))
    if hasattr(nodo, 'der'):
        nodos.extend(obtener_nodos(nodo.der))
    return nodos

def reemplazar_nodo(arbol, nodo_viejo, nodo_nuevo):
    """Reemplaza recursivamente la primera instancia de nodo_viejo por nodo_nuevo."""
    if arbol is nodo_viejo:
        return copy.deepcopy(nodo_nuevo)
    
    if hasattr(arbol, 'hijo'):
        if arbol.hijo is nodo_viejo:
            arbol.hijo = copy.deepcopy(nodo_nuevo)
        else:
            arbol.hijo = reemplazar_nodo(arbol.hijo, nodo_viejo, nodo_nuevo)
            
    if hasattr(arbol, 'izq'):
        if arbol.izq is nodo_viejo:
            arbol.izq = copy.deepcopy(nodo_nuevo)
        else:
            arbol.izq = reemplazar_nodo(arbol.izq, nodo_viejo, nodo_nuevo)
            
    if hasattr(arbol, 'der'):
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
def worker_evaluar(args):
    """
    Función que evalúa a un individuo (árbol) sobre el dataset proporcionado.
    Calcula el 'Balanced Accuracy' incorporando un MARGEN.
    Las áreas de clasificación se definen por:
      - f(x, y) > margen     -> Asignado a la Clase 1
      - f(x, y) < -margen    -> Asignado a la Clase 0
    Si cae en el medio, se cuenta como fallo al no superar la zona de incertidumbre.
    """
    individuo, dataset = args
    tp = tn = fp = fn = 0
    margen = 0.1
    
    for x, y, true_class in dataset:
        try:
            res = individuo.evaluar(x, y)
            import math
            if math.isnan(res) or math.isinf(res):
                continue
            
            # Evaluamos según el margen
            if res > margen:
                pred_class = 1
            elif res < -margen:
                pred_class = 0
            else:
                pred_class = -1 # Indecisión / Dentro de la frontera
            
            # Comparar con el modelo que da el profesor
            if true_class == 1:
                if pred_class == 1: tp += 1
                else: fn += 1
            elif true_class == 0:
                if pred_class == 0: tn += 1
                else: fp += 1
        except Exception:
            pass
            
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (tpr + tnr) / 2
    
    # Penalización
    tamano = len(obtener_nodos(individuo))
    fitness = balanced_acc - (tamano * 0.001)
    
    return fitness if fitness > 0 else 0.0

# ==========================================
# CLASE MODELO CAJA NEGRA
# ==========================================
class BlackBoxModel:
    def __init__(self, path="blackbox_model.pkl"):
        self.model = joblib.load(path)
        
    def predict(self, X):
        return self.model.predict(X)

# ==========================================
# ALGORITMO GENÉTICO
# ==========================================
class AlgoritmoGenetico:
    def __init__(self, path_modelo, tam_poblacion=20, tam_elite=5, generaciones=50, prob_cruce=0.8, prob_mutacion=0.2, profundidad_inicial=4):
        self.path_modelo = path_modelo
        self.tam_poblacion = tam_poblacion
        self.tam_elite = tam_elite
        self.generaciones = generaciones
        self.prob_cruce = prob_cruce
        self.prob_mutacion = prob_mutacion
        self.profundidad_inicial = profundidad_inicial
        
        # Generar dataset de entrenamiento con puntos aleatorios usando el modelo caja negra
        print(f"Generando dataset exploratorio para {path_modelo}...")
        bb = BlackBoxModel(path_modelo)
        
        # Generar puntos de manera más distribuida para encontrar ambas clases
        X_points = []
        Y_labels = []
        
        # Muestreo en grid para capturar mejor las fronteras
        rango = np.linspace(-10, 10, 30)
        for i in rango:
            for j in rango:
                X_points.append([i, j])
                
        X_points = np.array(X_points)
        Y_labels = bb.predict(X_points)
        
        self.dataset = [(x[0], x[1], y) for x, y in zip(X_points, Y_labels)]
        clases, counts = np.unique(Y_labels, return_counts=True)
        print(f"Dataset generado. Distribución de clases: {dict(zip(clases, counts))}")

    def inicializar_poblacion(self):
        """Inicializa la población de árboles con individuos aleatorios y calcula su fitness."""
        poblacion = []
        for _ in range(self.tam_poblacion):
            arbol = generar_arbol_aleatorio(self.profundidad_inicial)
            poblacion.append((arbol, 0.0))  # El fitness inicial es 0.0
        return poblacion

    def evaluar_poblacion(self, poblacion):
        """Evalúa toda la población usando multiprocesamiento."""
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            args = [(individuo, self.dataset) for individuo, fitness in poblacion]
            resultados_fitness = pool.map(worker_evaluar, args)
            
        # Actualizamos la población con el fitness calculado (vector por pares: individuo, fitness)
        poblacion_evaluada = [(poblacion[i][0], resultados_fitness[i]) for i in range(len(poblacion))]
        
        # Ordenamos de mayor a menor fitness
        poblacion_evaluada.sort(key=lambda x: x[1], reverse=True)
        return poblacion_evaluada

    def seleccion_torneo(self, poblacion, k=3):
        """Selección por torneo para elegir padres."""
        seleccionados = random.sample(poblacion, k)
        seleccionados.sort(key=lambda x: x[1], reverse=True)
        return seleccionados[0][0] # Retorna el mejor individuo (árbol)

    def ejecutar(self):
        print("\n=== INICIANDO ALGORITMO GENÉTICO ===")
        # 1. Inicializar
        poblacion = self.inicializar_poblacion()
        
        # 2. Bucle Generacional
        for gen in range(self.generaciones):
            # Evaluar
            poblacion = self.evaluar_poblacion(poblacion)
            
            mejor_fitness = poblacion[0][1]
            print(f"Generación {gen+1}/{self.generaciones} | Mejor Fitness (Accuracy): {mejor_fitness:.4f}")
            
            # Si encontramos el modelo perfecto
            if mejor_fitness >= 0.99:
                print("¡Solución óptima alcanzada!")
                break
                
            nueva_poblacion = []
            
            # 3. Elitismo (Guardar los mejores)
            elite = poblacion[:self.tam_elite]
            nueva_poblacion.extend([(ind, 0.0) for ind, fit in elite]) # Reseteamos fitness para la sig generacion
            
            # 4. Cruzamiento y Mutación para rellenar la población
            while len(nueva_poblacion) < self.tam_poblacion:
                padre1 = self.seleccion_torneo(poblacion)
                padre2 = self.seleccion_torneo(poblacion)
                
                if random.random() < self.prob_cruce:
                    hijo1, hijo2 = cruzar_arboles(padre1, padre2)
                else:
                    hijo1, hijo2 = copy.deepcopy(padre1), copy.deepcopy(padre2)
                    
                if random.random() < self.prob_mutacion:
                    hijo1 = mutar_arbol(hijo1)
                if random.random() < self.prob_mutacion:
                    hijo2 = mutar_arbol(hijo2)
                    
                nueva_poblacion.append((hijo1, 0.0))
                if len(nueva_poblacion) < self.tam_poblacion:
                    nueva_poblacion.append((hijo2, 0.0))
                    
            poblacion = nueva_poblacion

        # Evaluación final
        poblacion = self.evaluar_poblacion(poblacion)
        mejor_individuo = poblacion[0]
        print("\n=== FIN DEL ALGORITMO ===")
        print(f"Mejor Fitness Final: {mejor_individuo[1]:.4f}")
        print(f"Ecuación Encontrada para la Frontera:\n{mejor_individuo[0]}")
        return mejor_individuo

if __name__ == '__main__':
    # Ejecutamos el genético para el modelo A
    print("\n----------------------------------------------------")
    print("EJECUTANDO GENÉTICO PARA: blackbox_modelA.pkl")
    print("----------------------------------------------------")
    # Los parámetros: 20 individuos, 5 élite según las especificaciones
    ag_A = AlgoritmoGenetico("blackbox_modelA.pkl", tam_poblacion=20, tam_elite=5, generaciones=20)
    mejor_A = ag_A.ejecutar()

    # Ejecutamos el genético para el modelo B
    print("\n----------------------------------------------------")
    print("EJECUTANDO GENÉTICO PARA: blackbox_modelB.pkl")
    print("----------------------------------------------------")
    ag_B = AlgoritmoGenetico("blackbox_modelB.pkl", tam_poblacion=20, tam_elite=5, generaciones=20)
    mejor_B = ag_B.ejecutar()
