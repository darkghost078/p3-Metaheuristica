
from arbol import generar_arbol_aleatorio, Nodo
import numpy as np
from scipy.optimize import fsolve

def encontrar_posibles_y(arbol, x_fija, y_min=-5.0, y_max=5.0, num_semillas=20):
    """
    Dada una x fija, encuentra los posibles valores de y que hacen que f(x,y) = 0.
    """
    # 1. Definimos la función envolvente donde 'x' es constante
    def funcion_objetivo(y_test):
        # fsolve envía un array, extraemos el primer valor
        y_val = y_test[0] if isinstance(y_test, np.ndarray) else y_test
        return arbol.evaluar(x_fija, y_val)

    posibles_y = []
    
    # 2. Generamos puntos de partida (semillas) entre un rango
    # Esto es necesario porque la ecuación puede tener múltiples raíces (ej. si usa Seno)
    semillas = np.linspace(y_min, y_max, num_semillas)
    
    for semilla in semillas:
        try:
            # fsolve busca numéricamente el punto donde la función da 0
            y_sol, info, ier, msg = fsolve(funcion_objetivo, semilla, full_output=True)
            
            # ier == 1 significa que el algoritmo convergió a una solución
            if ier == 1:
                y_encontrada = y_sol[0]
                
                # Verificamos que el resultado evaluado sea realmente cercano a 0 (evitar falsos positivos)
                if abs(funcion_objetivo(y_encontrada)) < 1e-4:
                    
                    # Evitamos guardar raíces duplicadas
                    es_nueva = True
                    for py in posibles_y:
                        if abs(y_encontrada - py) < 1e-3:
                            es_nueva = False
                            break
                            
                    if es_nueva:
                        posibles_y.append(y_encontrada)
        except Exception:
            # Ignoramos errores de evaluación (como divisiones extremas) y continuamos
            continue
            
    return posibles_y
if __name__ == "__main__":
    print("--- Generador de Fronteras Aleatorias ---")
    
    # Generamos un árbol con una profundidad máxima de 3 niveles
    profundidad = 3
    mi_frontera = generar_arbol_aleatorio(profundidad)
    
    print("\n1. Ecuación generada:")
    print(mi_frontera)
    
    # === NUEVA LÓGICA AQUÍ ===
    x_prueba = 2.0
    print(f"\n2. Buscando valores de 'y' para la frontera cuando x = {x_prueba}")
    
    valores_y = encontrar_posibles_y(mi_frontera, x_prueba)
    
    if valores_y:
        print(f"Se encontraron las siguientes soluciones para y: {valores_y}")
        # Comprobación rápida:
        for y_sol in valores_y:
            res = mi_frontera.evaluar(x_prueba, y_sol)
            print(f" -> Comprobando f({x_prueba}, {y_sol:.4f}) = {res:.6f} (debería ser ~0)")
    else:
        print("No se encontraron valores de 'y' en el rango buscado (la frontera no pasa por esa 'x' o no intersecta el eje 0).")