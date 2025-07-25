def aux_motors(aux_engine_activations, roll):
    """
    Calculates the agent's reward based on:
    1. Efficient use of auxiliary engines (left/right)
    2. Angular stability (low roll)
    3. Improvement or regression compared to previous score
    
    Calcula la recompensa del agente en base a:
    1. Uso eficiente de los motores auxiliares (izquierdo/derecho)
    2. Estabilidad angular (roll bajo)
    3. Mejora o regresión con respecto a la puntuación anterior
    """

    # Sum lateral activations
    # Sumar activaciones laterales
    total_aux_activations = aux_engine_activations['left'] + aux_engine_activations['right']

    # 1. Penalty for excessive use of lateral engines
    # Penalizamos solo si activaciones > tolerancia (ej. 30)
    tolerance = 30
    if total_aux_activations > tolerance:
        # Penalize only activations beyond tolerance threshold
        # Penaliza solo activaciones más allá del umbral de tolerancia
        overuse_penalty = -(total_aux_activations - tolerance) * 500
    else:
        overuse_penalty = 0

    # 2. Penalty for inclination (ideal: roll near 0)
    # El roll puede ir de -π a π. Penaliza más conforme se aleja de 0.
    # Absolute value of roll (deviation from vertical) multiplied by penalty factor
    # Valor absoluto del roll (desviación de la vertical) multiplicado por factor de penalización
    stability_penalty = -abs(roll) * 1000

    # 3. Base points sum
    # Suma base de puntos
    points = int((overuse_penalty + stability_penalty) / 100)

    return {
        'reward': points
    }