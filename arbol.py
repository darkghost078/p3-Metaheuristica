import math
import random

# ==========================================
# 1. DEFINICIÓN DE CLASES (AST)
# ==========================================
class Nodo:
    def evaluar(self, x, y):
        raise NotImplementedError()

# --- Terminales (Hojas) ---
class Variable(Nodo):
    def __init__(self, nombre):
        self.nombre = nombre
    def evaluar(self, x, y):
        return x if self.nombre == 'x' else y
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
            return 1.0 # Protección
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
            return Variable(random.choice(['x', 'y']))
        else:
            return Constante()
            
    # CASO RECURSIVO: Elegimos qué tipo de nodo crear
    # Usamos pesos para favorecer que el árbol crezca un poco antes de cerrarse prematuramente
    opciones = ['terminal', 'binario']
    pesos = [1, 5] 
    tipo_nodo = random.choices(opciones, weights=pesos)[0]
    
    if tipo_nodo == 'terminal':
        if random.random() < 0.5:
            return Variable(random.choice(['x', 'y']))
        else:
            return Constante()
            
        
    else: # binario
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
    
    def distancia_cuadrada(vars):
        x, y = vars
        return (x - x0)**2 + (y - y0)**2
        
    def restriccion_frontera(vars):
        x, y = vars
        try:
            return arbol.evaluar(x, y)
        except Exception:
            return 1e9
            
    restricciones = [{'type': 'eq', 'fun': restriccion_frontera}]
    
    punto_inicial = np.array([x0, y0])
    
    resultado = minimize(
        distancia_cuadrada, 
        punto_inicial, 
        method='SLSQP', 
        constraints=restricciones,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    
    if resultado.success:
        x_opt, y_opt = resultado.x
        
        if abs(arbol.evaluar(x_opt, y_opt)) < 1e-3:
            return (x_opt, y_opt)
        else:
            print("El optimizador terminó, pero el punto no está exactamente en la frontera.")
            return (x_opt, y_opt)
    else:
        print(f"No se pudo encontrar un punto convergente: {resultado.message}")
        return None

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