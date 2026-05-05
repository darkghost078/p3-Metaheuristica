from arbol import generar_arbol_aleatorio, Nodo
import numpy as np
from scipy.optimize import minimize

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
        return (x - x0)**2 + (y - y0)**2
        
    # 2. Restricción: El punto debe evaluar a 0 en tu árbol de gramática
    def restriccion_frontera(vars):
        x, y = vars
        # Protegemos contra posibles desbordamientos matemáticos
        try:
            return arbol.evaluar(x, y)
        except Exception:
            return 1e9 # Penalización alta si la evaluación falla
            
    # Definimos la restricción de igualdad (eq): f(x, y) == 0 para scipy
    restricciones = [{'type': 'eq', 'fun': restriccion_frontera}]
    
    # 3. Optimización
    # Usamos el propio punto de origen como semilla inicial de búsqueda
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
        
        # Verificación final para asegurar que realmente estamos en la frontera
        if abs(arbol.evaluar(x_opt, y_opt)) < 1e-3:
            return (x_opt, y_opt)
        else:
            print("El optimizador terminó, pero el punto no está exactamente en la frontera.")
            return (x_opt, y_opt)
    else:
        print(f"No se pudo encontrar un punto convergente: {resultado.message}")
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
    Genera puntos a lo largo de la frontera f(x,y)=0 usando la recta tangente 
    como guía y proyectándolos para que sean (aproximadamente) equidistantes.
    """
    x_c, y_c = punto_centro
    puntos_frontera = []
    
    # 1. Obtenemos el vector gradiente en el centro
    gx, gy = calcular_gradiente(arbol, x_c, y_c)
    norma = np.hypot(gx, gy) # Equivalente a sqrt(gx^2 + gy^2)
    
    if norma < 1e-8:
        print("Advertencia: El gradiente es casi nulo. Usando dirección horizontal.")
        tx, ty = 1.0, 0.0
    else:
        # 2. Calculamos el vector tangente unitario (perpendicular al gradiente)
        tx, ty = -gy / norma, gx / norma

    # 3. Generamos distancias equidistantes centradas en 0
    # Ejemplo para 100 puntos: vamos desde una distancia -D hasta +D
    distancias_t = np.linspace(-paso * (num_puntos // 2), paso * (num_puntos // 2), num_puntos)
    
    for t in distancias_t:
        # Si estamos exactamente en el centro, simplemente agregamos el punto original
        if abs(t) < 1e-6:
            puntos_frontera.append((x_c, y_c))
            continue
            
        # Semilla inicial sobre la recta tangente
        x_guess = x_c + t * tx
        y_guess = y_c + t * ty
        
        # 4. Proyectamos la semilla a la frontera real usando tu optimizador
        punto_proyectado = encontrar_punto_mas_cercano(arbol, (x_guess, y_guess))
        
        if punto_proyectado:
            puntos_frontera.append(punto_proyectado)
            
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
            
        tuplas_punto = []
        
        # 3. Generamos los pares de puntos para cada distancia
        for d in distancias:
            # Punto a un lado (sumando el vector normal)
            p_mas = (x + d * nx, y + d * ny)
            # Punto al otro lado (restando el vector normal)
            p_menos = (x - d * nx, y - d * ny)
            
            # Guardamos la tupla con ambos puntos
            tuplas_punto.append((p_mas, p_menos))
            
        vector_ortogonales.append(tuplas_punto)
        
    return vector_ortogonales
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("--- PRUEBA DE PUNTOS ORTOGONALES ---")
    
    # 1. Generar frontera aleatoria
    profundidad = 4
    mi_frontera = generar_arbol_aleatorio(profundidad)
    print("\n1. Ecuación generada:")
    print(mi_frontera)
    
    # 2. Encontrar punto central y vecindad
    punto_origen = (2.0, -1.5)
    punto_central = encontrar_punto_mas_cercano(mi_frontera, punto_origen)
    
    if punto_central:
        print(f"\n2. Generando 100 puntos en la frontera a partir de {punto_central}...")
        puntos_frontera = generar_vecindad_frontera(mi_frontera, punto_central, num_puntos=100, paso=0.05)
        
        # 3. Generar los puntos ortogonales
        # Usamos distancias de 0.1, 0.2 y 0.3
        distancias_prueba = [0.1, 0.2, 0.3]
        print(f"\n3. Generando 3 tuplas de puntos ortogonales por cada punto de la frontera...")
        vector_100 = generar_puntos_ortogonales(mi_frontera, puntos_frontera, distancias=distancias_prueba)
        
        print(f"   -> Tamaño del vector resultante: {len(vector_100)} (Debería ser 100)")
        print(f"   -> Tamaño de elementos internos: {len(vector_100[0])} tuplas por punto")
        
        # 4. Visualización
        print("\n4. Dibujando... (Cierra la gráfica para finalizar)")
        plt.figure(figsize=(10, 10))
        
        # Dibujar la frontera principal
        X_front = [p[0] for p in puntos_frontera]
        Y_front = [p[1] for p in puntos_frontera]
        plt.plot(X_front, Y_front, 'b-', label='Frontera f(x,y)=0', linewidth=2)
        
        # Colores para las distintas distancias (rojo, verde, naranja)
        colores = ['r', 'g', 'orange']
        
        # Dibujar los puntos ortogonales
        # Extraemos los datos del vector_100 para plotearlos
        for idx_distancia, d in enumerate(distancias_prueba):
            x_mas, y_mas = [], []
            x_menos, y_menos = [], []
            
            for tuplas in vector_100:
                p_m, p_me = tuplas[idx_distancia]
                x_mas.append(p_m[0])
                y_mas.append(p_m[1])
                x_menos.append(p_me[0])
                y_menos.append(p_me[1])
                
            plt.scatter(x_mas, y_mas, color=colores[idx_distancia], s=10, alpha=0.6, label=f'Distancia +{d}')
            plt.scatter(x_menos, y_menos, color=colores[idx_distancia], s=10, alpha=0.6, marker='x', label=f'Distancia -{d}')

        plt.plot(punto_origen[0], punto_origen[1], 'k*', markersize=12, label='Punto Origen')
        
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid(color='gray', linestyle='--', linewidth=0.2)
        plt.legend()
        plt.title("Puntos Ortogonales a la Frontera Generada")
        plt.xlabel("Eje X")
        plt.ylabel("Eje Y")
        plt.axis('equal') # Fundamental para que las proporciones ortogonales se vean reales
        plt.savefig("Omar_Sadiq.png")
    else:
        print("No se encontró un punto de convergencia. Ejecuta de nuevo.")
