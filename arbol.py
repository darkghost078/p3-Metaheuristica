import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# ==========================================
# 1. DEFINICIÓN DE CLASES (AST)
# ==========================================
class Nodo:
    def evaluar(self, x, y):
        raise NotImplementedError()

    def graf(self, limite_inferior=-5.0, limite_superior=5.0, resolucion=200):
        """
        Grafica la frontera f(x,y) = 0 y colorea la región positiva.
        """
        # 1. Generación de los ejes espaciales
        x_vals = np.linspace(limite_inferior, limite_superior, resolucion)
        y_vals = np.linspace(limite_inferior, limite_superior, resolucion)

        # 2. Creación de la malla bidimensional
        X, Y = np.meshgrid(x_vals, y_vals)

        # 3. Vectorización y evaluación
        evaluar_vectorizado = np.vectorize(self.evaluar)
        Z = evaluar_vectorizado(X, Y)

        # 4. Configuración del entorno 2D
        fig, ax = plt.subplots(figsize=(8, 8))

        # --- CAMBIOS PRINCIPALES ---

        # A. Colorear la región positiva
        # Usamos niveles desde 0 hasta el máximo de Z para rellenar
        ax.contourf(
            X,
            Y,
            Z,
            levels=[0, Z.max() if Z.max() > 0 else 1],
            colors=["#e6f3ff"],
            alpha=0.5,
        )

        # B. Dibujar la línea de la frontera (f(x,y) = 0)
        ax.contour(X, Y, Z, levels=[0], colors="red", linewidths=2)

        # ---------------------------

        # 6. Etiquetas y formato
        ax.set_xlabel("Eje X")
        ax.set_ylabel("Eje Y")
        ax.set_title(f"Frontera de decisión y región positiva\n{str(self)}")

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)


# --- Terminales (Hojas) ---
class Variable(Nodo):
    def __init__(self, nombre):
        self.nombre = nombre

    def evaluar(self, x, y):
        return x if self.nombre == "x" else y

    def __str__(self):
        return self.nombre


class Constante(Nodo):
    def __init__(self, valor=None):
        self.valor = valor if valor is not None else random.uniform(-5.0, 5.0)

    def evaluar(self, x, y):
        return self.valor

    def __str__(self):
        return f"{self.valor:.2f}"


# --- Operadores Binarios ---
class Suma(Nodo):
    def __init__(self, izq, der):
        self.izq, self.der = izq, der

    def evaluar(self, x, y):
        return self.izq.evaluar(x, y) + self.der.evaluar(x, y)

    def __str__(self):
        return f"({self.izq} + {self.der})"


class Resta(Nodo):
    def __init__(self, izq, der):
        self.izq, self.der = izq, der

    def evaluar(self, x, y):
        return self.izq.evaluar(x, y) - self.der.evaluar(x, y)

    def __str__(self):
        return f"({self.izq} - {self.der})"


class Multiplicacion(Nodo):
    def __init__(self, izq, der):
        self.izq, self.der = izq, der

    def evaluar(self, x, y):
        return self.izq.evaluar(x, y) * self.der.evaluar(x, y)

    def __str__(self):
        return f"({self.izq} * {self.der})"


class DivisionProtegida(Nodo):
    def __init__(self, izq, der):
        self.izq, self.der = izq, der

    def evaluar(self, x, y):
        den = self.der.evaluar(x, y)
        if abs(den) < 1e-6:
            return 1e9
        return self.izq.evaluar(x, y) / den

    def __str__(self):
        return f"({self.izq} / {self.der})"


# ==========================================
# 2. GENERADOR ALEATORIO
# ==========================================
def generar_arbol_aleatorio(profundidad_maxima):
    """
    Genera un árbol de sintaxis abstracta de forma recursiva.
    """
    # CASO BASE: Si llegamos al límite de profundidad, debemos devolver un terminal (hoja)
    if profundidad_maxima == 0:
        if random.random() < 0.5:
            return Variable(random.choice(["x", "y"]))
        else:
            return Constante()

    # CASO RECURSIVO: Elegimos qué tipo de nodo crear
    # Usamos pesos para favorecer que el árbol crezca un poco antes de cerrarse prematuramente
    opciones = ["terminal", "binario"]
    pesos = [1, 5]
    tipo_nodo = random.choices(opciones, weights=pesos)[0]

    if tipo_nodo == "terminal":
        if random.random() < 0.5:
            return Variable(random.choice(["x", "y"]))
        else:
            return Constante()

    else:  # binario
        # Generamos los dos hijos recursivamente
        hijo_izq = generar_arbol_aleatorio(profundidad_maxima - 1)
        hijo_der = generar_arbol_aleatorio(profundidad_maxima - 1)

        # Elegimos un operador binario al azar
        Operador = random.choice([Suma, Resta, Multiplicacion, DivisionProtegida])
        return Operador(hijo_izq, hijo_der)


def encontrar_punto_mas_cercano(arbol, punto_origen):
    """
    Dado un árbol y un punto de origen (x0, y0), encuentra el punto (x, y)
    más cercano que pertenezca a la frontera f(x, y) = 0.
    """
    x0, y0 = punto_origen

    # 1. Función objetivo: Minimizar la distancia al cuadrado
    # Es más eficiente computacionalmente minimizar la distancia al cuadrado que la raíz cuadrada
    def distancia_cuadrada(vars):
        x, y = vars
        return (x - x0) ** 2 + (y - y0) ** 2

    # 2. Restricción: El punto debe evaluar a 0 en tu árbol de gramática
    def restriccion_frontera(vars):
        x, y = vars
        # Protegemos contra posibles desbordamientos matemáticos
        try:
            return arbol.evaluar(x, y)
        except Exception:
            return 1e9  # Penalización alta si la evaluación falla

    # Definimos la restricción de igualdad (eq): f(x, y) == 0 para scipy
    restricciones = [{"type": "eq", "fun": restriccion_frontera}]

    # 3. Optimización
    # Usamos el propio punto de origen como semilla inicial de búsqueda
    punto_inicial = np.array([x0, y0])

    resultado = minimize(
        distancia_cuadrada,
        punto_inicial,
        method="SLSQP",
        constraints=restricciones,
        options={"maxiter": 1000, "ftol": 1e-6},
    )

    if resultado.success:
        x_opt, y_opt = resultado.x

        # Verificación final para asegurar que realmente estamos en la frontera
        if abs(arbol.evaluar(x_opt, y_opt)) < 1e-3:
            return (x_opt, y_opt)
        else:
            print(
                "El optimizador terminó, pero el punto no está exactamente en la frontera."
            )
            return (x_opt, y_opt)
    else:
        # print(f"No se pudo encontrar un punto convergente: {resultado.message}")
        return None


def calcular_gradiente(arbol, x, y, h=1e-5):
    """
    Calcula el gradiente numérico (derivadas parciales) en un punto dado.
    """
    df_dx = (arbol.evaluar(x + h, y) - arbol.evaluar(x - h, y)) / (2 * h)
    df_dy = (arbol.evaluar(x, y + h) - arbol.evaluar(x, y - h)) / (2 * h)
    return df_dx, df_dy


def generar_vecindad_frontera(arbol, punto_centro, num_puntos=100, paso=0.05):
    """
    Genera puntos iterativamente a lo largo de la frontera f(x,y)=0.
    Sigue la tangente local paso a paso y proyecta hacia la curva real.
    """
    mitad = num_puntos // 2

    def avanzar_sobre_curva(punto_inicial, direccion, pasos):
        puntos = []
        pto_actual = punto_inicial

        tx_prev, ty_prev = 1.0, 0.0

        for _ in range(pasos):
            x_c, y_c = pto_actual
            gx, gy = calcular_gradiente(arbol, x_c, y_c)
            norma = np.hypot(gx, gy)

            if np.isnan(norma) or np.isinf(norma) or norma > 1e6:
                tx, ty = tx_prev, ty_prev
            elif norma < 1e-8:
                tx, ty = 1.0, 0.0
            else:
                tx, ty = -gy / norma, gx / norma
                tx_prev, ty_prev = tx, ty

            if direccion < 0:
                tx, ty = -tx, -ty

            x_guess = x_c + paso * tx
            y_guess = y_c + paso * ty

            punto_proyectado = encontrar_punto_mas_cercano(arbol, (x_guess, y_guess))

            if punto_proyectado:
                puntos.append(punto_proyectado)
                pto_actual = punto_proyectado
            else:
                pto_actual = (x_c + paso * tx * 2.0, y_c + paso * ty * 2.0)

        return puntos

    puntos_positivos = avanzar_sobre_curva(punto_centro, 1, mitad)
    puntos_negativos = avanzar_sobre_curva(punto_centro, -1, mitad)

    puntos_frontera = puntos_negativos[::-1] + [punto_centro] + puntos_positivos

    return puntos_frontera


def generar_puntos_ortogonales(arbol, puntos_frontera, distancias=[0.1, 0.2, 0.3]):
    """
    Dado un conjunto de puntos en la frontera f(x,y)=0, genera pares de puntos
    ortogonales (uno a cada lado) a distancias incrementales.
    Retorna una lista de la misma longitud que puntos_frontera.
    """
    vector_ortogonales = []

    for x, y in puntos_frontera:
        # 1. Calculamos el gradiente en el punto de la frontera
        gx, gy = calcular_gradiente(arbol, x, y)
        norma = np.hypot(gx, gy)

        # 2. Vector normal unitario (apunta perpendicular a la frontera)
        if norma < 1e-8:
            # Si la derivada es 0 (ej. en una zona plana o constante), usamos un vector por defecto
            nx, ny = 0.0, 1.0
        else:
            nx, ny = gx / norma, gy / norma

        # 3. Generamos los pares de puntos para cada distancia
        for d in distancias:
            # Punto a un lado (sumando el vector normal)
            p_mas = (x + d * nx, y + d * ny)
            # Punto al otro lado (restando el vector normal)
            p_menos = (x - d * nx, y - d * ny)

            # Guardamos la tupla con ambos puntos
            vector_ortogonales.append((p_mas, p_menos))

    return vector_ortogonales


def altura_arbol(nodo):
    if isinstance(nodo, (Variable, Constante)):
        return 1
    altura_izq = altura_arbol(nodo.izq) if hasattr(nodo, "izq") else 0
    altura_der = altura_arbol(nodo.der) if hasattr(nodo, "der") else 0
    return 1 + max(altura_izq, altura_der)


def simplificar(nodo):
    if isinstance(nodo, (Variable, Constante)):
        return nodo

    if hasattr(nodo, "izq"):
        nodo.izq = simplificar(nodo.izq)
    if hasattr(nodo, "der"):
        nodo.der = simplificar(nodo.der)

    if hasattr(nodo, "izq") and hasattr(nodo, "der"):
        if isinstance(nodo.izq, Constante) and isinstance(nodo.der, Constante):
            try:
                val = nodo.evaluar(0, 0)
                return Constante(val)
            except Exception:
                pass

    if isinstance(nodo, Suma):
        if isinstance(nodo.der, Constante) and abs(nodo.der.valor) < 1e-6:
            return nodo.izq
        if isinstance(nodo.izq, Constante) and abs(nodo.izq.valor) < 1e-6:
            return nodo.der

    elif isinstance(nodo, Resta):
        if isinstance(nodo.der, Constante) and abs(nodo.der.valor) < 1e-6:
            return nodo.izq

    elif isinstance(nodo, Multiplicacion):
        if isinstance(nodo.der, Constante):
            if abs(nodo.der.valor - 1.0) < 1e-6:
                return nodo.izq
            if abs(nodo.der.valor) < 1e-6:
                return Constante(0.0)
        if isinstance(nodo.izq, Constante):
            if abs(nodo.izq.valor - 1.0) < 1e-6:
                return nodo.der
            if abs(nodo.izq.valor) < 1e-6:
                return Constante(0.0)

    elif isinstance(nodo, DivisionProtegida):
        if isinstance(nodo.der, Constante) and abs(nodo.der.valor - 1.0) < 1e-6:
            return nodo.izq
        if isinstance(nodo.izq, Constante) and abs(nodo.izq.valor) < 1e-6:
            return Constante(0.0)

    return nodo


# ==========================================
# 3. BLOQUE MAIN
# ==========================================
if __name__ == "__main__":
    print("--- Generador de Fronteras Aleatorias ---")

    # Generamos un árbol con una profundidad máxima de 3 niveles
    profundidad = 3
    mi_frontera = generar_arbol_aleatorio(profundidad)

    print("\n1. Ecuación generada:")
    print(mi_frontera)

    print("\n2. Evaluando un punto de prueba: P(x=2.0, y=-1.5)")
    try:
        resultado = mi_frontera.evaluar(2.0, -1.5)
        print(f"Resultado de f(2.0, -1.5) = {resultado:.4f}")

        # Lógica de clasificación
        if resultado > 0:
            print("-> El punto pertenece a la CLASE A (Positiva)")
        else:
            print("-> El punto pertenece a la CLASE B (Negativa)")

    except Exception as e:
        print(f"Ocurrió un error al evaluar: {e}")

