import numpy as np

def safe_crash(state, epsilon):
    """
    Positive reinforcement for soft crashes (safe landings).
    Scores from 0 to 100 based on impact softness (closer to 0 is better)
    and exploration level (epsilon).
    Also returns a soft_landing flag: 1 if considered safe, 0 otherwise.

    Refuerzo positivo por choques suaves (aterrizajes seguros).
    Puntúa de 0 a 100 dependiendo de la suavidad del impacto (más cerca de 0 es mejor)
    y del nivel de exploración (epsilon).
    Devuelve también un flag soft_landing: 1 si se considera seguro, 0 si no.
    """
    # Check if legs are in contact with ground / Verificar si las patas están en contacto con el suelo
    legs_contact = state[6] == 1 or state[7] == 1
    
    if not legs_contact:
        return {'reward': 0, 'soft_crash': 0}

    # Get velocity components / Obtener componentes de velocidad
    vx = float(state[2])
    vy = float(state[3])

    # Calculate impact intensity / Calcular intensidad del impacto
    raw_impact = (np.abs(vx) + np.abs(vy)) / 2
    impact = min(1.0, raw_impact)  # Cap at 1.0 / Límite máximo de 1.0

    # Calculate safety score / Calcular puntuación de seguridad
    safety_score = (1.0 - impact) ** 2  # Quadratic scaling / Escalado cuadrático
    reward = 100 * safety_score * epsilon  # Scale and adjust by exploration / Escalar y ajustar por exploración

    # Determine if it's a soft crash (impact < 0.3) / Determinar si es choque suave (impacto < 0.3)
    is_soft_crash = 1 if impact < 0.3 else 0

    return {
        'reward': round(reward),
        'soft_crash': is_soft_crash
    }