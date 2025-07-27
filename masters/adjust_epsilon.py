import numpy as np

def adjust_epsilon(epsilon, points, last_points,
                   min_epsilon=0.001, max_epsilon=1.0,
                   decay_base=0.01, increase_base=0.01,
                   sensitivity=500.0):
    """
    Adjusts exploration rate (epsilon) based on performance changes.
    Uses different strategies depending on current exploration level.
    
    Ajusta la tasa de exploración (epsilon) basado en cambios de rendimiento.
    Usa diferentes estrategias según el nivel actual de exploración.
    """   
    # Calculate performance difference between episodes
    # Calcular diferencia de rendimiento entre episodios
    delta = points - last_points

    # Normalize adjustment using hyperbolic tangent
    # Normalizar ajuste usando tangente hiperbólica
    adjustment = np.tanh(delta / sensitivity)  # Range: [-1, 1] / Rango: [-1, 1]

    # High exploration phase (epsilon > 0.5)
    # Fase de alta exploración (epsilon > 0.5)
    if epsilon > 0.5:
        # When performance improves (delta > 0)
        # Cuando el rendimiento mejora (delta > 0)
        if delta > 0:
            # Apply stronger decay to reduce exploration
            # Aplicar decaimiento fuerte para reducir exploración
            decay = (decay_base * 1) * adjustment * (epsilon / max_epsilon)
            epsilon -= decay
        else:
            # Apply weaker increase when performance worsens
            # Aplicar incremento débil cuando el rendimiento empeora
            increase = (increase_base * 0.5) * abs(adjustment) * ((max_epsilon - epsilon) / max_epsilon)
            epsilon += increase
    # Low exploration phase (epsilon <= 0.5)
    # Fase de baja exploración (epsilon <= 0.5)
    else:
        if delta > 0:
            # Apply weaker decay when improving
            # Aplicar decaimiento débil cuando mejora
            decay = (decay_base * 1) * adjustment * (epsilon / max_epsilon)
            epsilon -= decay
        else:
            # Apply stronger increase when worsening
            # Aplicar incremento fuerte cuando empeora
            increase = (increase_base * 0.5) * abs(adjustment) * ((max_epsilon - epsilon) / max_epsilon)
            epsilon += increase

    # Keep epsilon within defined bounds
    # Mantener epsilon dentro de límites definidos
    epsilon = max(min(epsilon, max_epsilon), min_epsilon)

    return round(epsilon, 3), points