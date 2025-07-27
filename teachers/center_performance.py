import math

def center_performance(state):
    """
    Calculates a reward between -100 and +100 based on distance to center point [0, 0].
    / Calcula una recompensa entre -100 y +100 según la distancia al punto central [0, 0].

    Args / Argumentos:
        state (list): Agent's state (uses state[0] = x, state[1] = y)
        / Estado del agente (usa state[0] = x, state[1] = y)

    Returns / Retorna:
        dict: Dictionary containing reward value
        / Diccionario conteniendo el valor de recompensa
    """
    # Get current position coordinates / Obtener coordenadas de posición actual
    x = state[0]
    y = state[1]

    # Calculate Euclidean distance to center / Calcular distancia euclidiana al centro
    distance = math.sqrt(x**2 + y**2)

    # Calculate scaled reward using exponential decay:
    # - Closer to center → higher positive reward (max +100)
    # - Farther from center → higher negative penalty (min -100)
    # - Factor 4 controls sensitivity (adjustable)
    # Calcular recompensa escalada usando decaimiento exponencial:
    # - Más cerca del centro → recompensa positiva mayor (máx +100)
    # - Más lejos del centro → penalización negativa mayor (mín -100)
    # - Factor 4 controla sensibilidad (ajustable)
    scaled = 200 * math.exp(-distance * 4) - 100  # ∈ (-100, +100)

    return {
        'reward': round(scaled, 2) 
    }